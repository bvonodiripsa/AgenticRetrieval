"""Standard scaffold with native function calling over Cosmos DB (vector + fulltext + ranker)."""
from tqdm import tqdm
import argparse, asyncio, json, re, time, yaml
from pathlib import Path
import httpx, openai as oai
import tiktoken
from openai import AsyncAzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
from azure.cosmos.aio import CosmosClient
from utils.fulltext import fulltext_search
from utils.ranker import rerank_documents

_enc = tiktoken.get_encoding("o200k_base")
def count_tokens(msgs):
    return sum(4 + len(_enc.encode(m["content"] if isinstance(m, dict) and "content" in m and isinstance(m["content"], str) else json.dumps(m) if isinstance(m, dict) else str(m))) for m in msgs) + 2

DEFAULT_MANAGEMENT_SCOPE = "https://management.azure.com/.default"


def _ranker_credential(rcfg):
    tenant_id = str(rcfg.get("tenant_id") or "").strip()
    return AzureCliCredential(tenant_id=tenant_id) if tenant_id else AzureCliCredential()


def _get_cli_token(rcfg, scope: str) -> str:
    return _ranker_credential(rcfg).get_token(scope).token


def build_ranker_url(rcfg) -> str:
    return f"https://{rcfg['account_name']}.{rcfg['region']}.dbinference.azure.com:443/inference/semanticReranking"

# --- Search helpers ---
async def embed(text):
    r = await embed_client.embeddings.create(input=[text], model=embed_cfg["embed_model"])
    dim = embed_cfg.get("embed_dimensions", 1536)
    raw = [float(x) for x in r.data[0].embedding]
    if len(raw) >= dim:
        return raw[:dim]
    return raw + [0.0] * (dim - len(raw))

_SAFE_FIELD_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$')

async def vec_search(container, emb, top_k, ef):
    if not _SAFE_FIELD_RE.match(ef):
        raise ValueError(f"Invalid embedding field name: {ef!r}")
    sql = f"SELECT TOP @k c, VectorDistance(c.{ef}, @emb) AS score FROM c ORDER BY VectorDistance(c.{ef}, @emb)"
    return [item.get("c", item) async for item in container.query_items(query=sql, parameters=[{"name":"@k","value":top_k},{"name":"@emb","value":emb}])]

async def rerank(query, docs, top_k):
    if not USE_RANKER or not docs:
        print("  [rerank] Reranker disabled or no docs to rerank, returning unranked results")
        return docs[-top_k:]
    indices = await rerank_documents(_r_http, _r_url, _r_tok, query, docs, top_k, _r_bs, _r_mr)
    if indices is None:
        print("  [rerank] Reranker failed, returning unranked results")
        return docs[:top_k]
    return [docs[i] for i in indices]

def fmt(doc):
    ex = {"_rid","_self","_etag","_attachments","_ts","_score","e"} | _all_embed
    return "\n".join(f"{k}: {v}" for k,v in doc.items() if k not in ex and v)

async def hyde_passage(query):
    """Generate a hypothetical answer passage for HyDE embedding."""
    prompt = f"Please write a passage to answer the question\nQuestion: {query}\nPassage:"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await llm.chat.completions.create(
                model=llm_cfg["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=512,
            )
            return r.choices[0].message.content or query
        except (oai.BadRequestError, oai.RateLimitError, oai.APIStatusError) as e:
            print(f"  [HyDE] LLM error ({attempt}/{MAX_RETRIES}): {e}")
            if attempt >= MAX_RETRIES:
                print(f"  [HyDE] Max retries exceeded, falling back to raw query")
                return query
            await asyncio.sleep(min(5 * 2 ** attempt, 60))

async def do_search(query, containers):
    if USE_HYDE:
        passage = await hyde_passage(query)
        print(f"  [HyDE] Generated passage for embedding: {passage[:100]}...")
        emb_hyde, emb_query = await asyncio.gather(embed(passage), embed(query))
        emb = [(a + b) / 2.0 for a, b in zip(emb_hyde, emb_query)]
    else:
        emb = await embed(query)
    tasks = []
    for sid, ret in _source_cfg.items():
        if sid in containers:
            tasks.append(vec_search(containers[sid], emb, ret["search_k"]*RERANK_MUL, _source_embed[sid]))
    for sid, fields in _source_ft.items():
        if sid in containers:
            tasks.append(fulltext_search(containers[sid], fields, query, _source_cfg[sid]["fulltext_search_k"]*RERANK_MUL))
    results = await asyncio.gather(*tasks)
    seen, all_d = set(), []
    for dl in results:
        for d in dl:
            did = d.get("id","")
            if did not in seen: seen.add(did); all_d.append(d)
    total_k = sum(r["search_k"]+r["fulltext_search_k"] for r in _source_cfg.values())
    texts = [fmt(d) for d in all_d]
    ranked_texts = await rerank(query, texts, total_k)
    # Map ranked texts back to source docs by index
    text_to_idx = {id(t): i for i, t in enumerate(texts)}
    ranked_indices = [text_to_idx[id(t)] for t in ranked_texts]
    return json.dumps([{"docid": all_d[i].get("id", ""), "text": ranked_texts[j]} for j, i in enumerate(ranked_indices)])

async def do_prune(docids, containers, doc_cache):
    parts = []
    for did in docids[:PRUNE_K]:
        if did in doc_cache:
            parts.append(f'<doc id="{did}">\n{doc_cache[did]}\n</doc>')
            continue
        found = False
        for container_id, c in containers.items():
            if found:
                break
            try:
                async for item in c.query_items(query="SELECT * FROM c WHERE c.id=@id", parameters=[{"name":"@id","value":did}]):
                    text = fmt(item)
                    doc_cache[did] = text
                    parts.append(f'<doc id="{did}">\n{text}\n</doc>')
                    found = True
                    break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"  [prune] Cosmos query failed for container={container_id}, docid={did}: {e}")
                continue
    return "Pruned context (only these documents remain):\n\n" + "\n\n".join(parts)

TOOLS = [
    {"type":"function","function":{"name":"initial_search","description":"Search the knowledge base using the original question. Must be called first and alone (no other tool calls in the same batch). Returns top results with docid and full document text.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"search","description":"Search knowledge base with a custom query. Returns top results with docid and full document text.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"prune","description":"Keep only the specified most relevant document IDs and discard all others from context. Use when context is large to free up space for more searches.","parameters":{"type":"object","properties":{"docids":{"type":"array","items":{"type":"string"},"description":"List of document IDs to keep"}},"required":["docids"]}}},
    {"type":"function","function":{"name":"find_information_gaps","description":"Identify information gaps in the retrieved documents that need to be addressed to answer the question. The tool can see all retrieved documents in the conversation. Returns a list of gaps to guide follow-up searches.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"final_answer","description":"Submit the final answer to the question. Must be called alone when you are ready to answer.","parameters":{"type":"object","properties":{"answer":{"type":"string","description":"The complete final answer to the question"}},"required":["answer"]}}},
]

async def process_question(q_obj, containers):
    t0 = time.perf_counter()
    query = q_obj["question_text"]
    qid = q_obj.get("question_id", "")
    print(f"\n{'='*60}\n[{qid}]: {query}\n{'='*60}")
    msgs = [{"role": "user", "content": QUERY_TEMPLATE.format(question=query)}]
    tc = {"initial_search": 0, "search": 0, "prune": 0, "find_information_gaps": 0, "final_answer": 0}
    doc_cache = {}
    initial_msg = msgs[0]
    retries = 0
    non_prune_rounds = 0
    for iteration in range(50):
        try:
            r = await llm.chat.completions.create(model=llm_cfg["llm_model"], messages=msgs, tools=TOOLS, temperature=llm_cfg.get("temperature", 0), max_completion_tokens=llm_cfg["max_completion_tokens"])
            retries = 0
        except (oai.BadRequestError, oai.RateLimitError, oai.APIStatusError) as e:
            retries += 1; print(f"  LLM error ({retries}/{MAX_RETRIES}): {e}")
            # On context overflow: roll back messages until under limit, then force prune
            if "context_length_exceeded" in str(e):
                print(f"  [auto-prune] Context overflow detected, rolling back messages...")
                # Remove messages from the end until we fit (keep at least initial_msg)
                while len(msgs) > 1 and count_tokens(msgs) > CONTEXT_LIMIT * 0.8:
                    msgs.pop()
                # Ensure we don't leave an orphaned assistant message with tool_calls
                # (the API requires tool response messages for every tool_call_id)
                while len(msgs) > 1:
                    last = msgs[-1]
                    if isinstance(last, dict) and last.get("role") == "assistant" and last.get("tool_calls"):
                        msgs.pop()
                    elif isinstance(last, dict) and last.get("role") == "tool":
                        msgs.pop()
                    else:
                        break
                # Force a prune call on next iteration
                msgs.append({"role": "user", "content": "CRITICAL: Context limit exceeded. You MUST call prune NOW to keep only the most relevant document IDs before doing anything else."})
                print(f"  [auto-prune] Rolled back to {count_tokens(msgs)} tokens, forcing prune")
                retries = 0  # reset retries since we changed the messages
                continue
            if retries >= MAX_RETRIES:
                elapsed = round(time.perf_counter() - t0, 2)
                print(f"  Max retries ({MAX_RETRIES}) exceeded, returning partial result")
                return {"question_id": qid, "query": query, "answer": "", "ground_truth": q_obj.get("answer",""),
                        "model": llm_cfg["llm_model"], "rounds": iteration+1, "elapsed_seconds": elapsed,
                        "tool_calls": tc, "error": f"Max retries exceeded: {e}"}
            await asyncio.sleep(min(5*2**retries, 300)); continue
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            answer = m.content or ""
            print(f"  Answer: {answer[:200]}...")
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  Elapsed: {elapsed}s")
            return {"question_id": qid, "query": query, "answer": answer, "ground_truth": q_obj.get("answer",""),
                    "model": llm_cfg["llm_model"], "rounds": iteration+1, "elapsed_seconds": elapsed,
                    "tool_calls": tc}
        for t in m.tool_calls:
            tc[t.function.name] = tc.get(t.function.name, 0) + 1
        # Handle final_answer tool call
        call_names = [t.function.name for t in m.tool_calls]
        if "final_answer" in call_names:
            # Enforce: final_answer must be the sole tool call in a turn
            if len(call_names) > 1:
                print(f"  [warn] final_answer mixed with other calls; returning error for non-final_answer calls")
                for t in m.tool_calls:
                    if t.function.name != "final_answer":
                        msgs.append({"role": "tool", "tool_call_id": t.id,
                                     "content": json.dumps({"error": "final_answer must be the only tool call in a turn; re-issue this call separately."})})
            fa_call = next(t for t in m.tool_calls if t.function.name == "final_answer")
            try:
                a = json.loads(fa_call.function.arguments)
                answer = a.get("answer", "")
            except (json.JSONDecodeError, TypeError):
                answer = ""
            # Append a tool response for final_answer to keep the conversation state valid
            msgs.append({"role": "tool", "tool_call_id": fa_call.id,
                         "content": json.dumps({"status": "ok"})})
            print(f"  Answer (via final_answer): {answer[:200]}...")
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"  Elapsed: {elapsed}s")
            return {"question_id": qid, "query": query, "answer": answer, "ground_truth": q_obj.get("answer",""),
                    "model": llm_cfg["llm_model"], "rounds": iteration+1, "elapsed_seconds": elapsed,
                    "tool_calls": tc}
        # Enforce: prune must be the sole tool call in a turn
        if "prune" in call_names and len(call_names) > 1:
            print(f"  [warn] prune mixed with other calls; returning error for non-prune calls")
            for t in m.tool_calls:
                if t.function.name != "prune":
                    msgs.append({"role": "tool", "tool_call_id": t.id,
                                 "content": json.dumps({"error": "prune must be the only tool call in a turn; re-issue this call separately."})})
            # Execute only the prune call
            prune_call = next(t for t in m.tool_calls if t.function.name == "prune")
            try:
                a = json.loads(prune_call.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                msgs.append({"role": "tool", "tool_call_id": prune_call.id,
                             "content": json.dumps({"error": f"Malformed tool arguments: {e}"})})
                continue
            out = await do_prune(a["docids"], containers, doc_cache)
            msgs.clear()
            msgs.append(initial_msg)
            msgs.append({"role": "assistant", "content": "I'll prune the context to focus on the most relevant documents."})
            msgs.append({"role": "user", "content": out})
            print(f"  [prune] Kept {len(a['docids'])} docs, context reset")
            continue
        async def _exec(t):
            try:
                a = json.loads(t.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                return t, json.dumps({"error": f"Malformed tool arguments: {e}"}), False
            if t.function.name == "initial_search":
                out = await do_search(query, containers)
                try:
                    for h in json.loads(out):
                        if h.get("text"):
                            doc_cache[h["docid"]] = h["text"]
                except asyncio.CancelledError: raise
                except Exception as e:
                    print(f"  [initial_search] failed to parse/cache search results: {e}")
            elif t.function.name == "search":
                out = await do_search(a["query"], containers)
                try:
                    for h in json.loads(out):
                        if h.get("text"):
                            doc_cache[h["docid"]] = h["text"]
                except asyncio.CancelledError: raise
                except Exception as e:
                    print(f"  [search] failed to parse/cache search results: {e}")
            elif t.function.name == "prune":
                out = await do_prune(a["docids"], containers, doc_cache)
                return t, out, True  # signal prune
            elif t.function.name == "find_information_gaps":
                # Ask the LLM to identify gaps using the current conversation context.
                # Retrieved documents live in tool-response messages, so we must surface
                # them to the gap-finder. Convert tool outputs into plain user messages
                # and drop the paired assistant tool_calls messages (which would be
                # orphaned without their tool responses and aren't useful as text).
                gap_msgs = []
                for mm in msgs:
                    if not isinstance(mm, dict):
                        gap_msgs.append(mm)
                        continue
                    role = mm.get("role")
                    if role == "assistant" and mm.get("tool_calls"):
                        # Skip assistant turns that only carry tool_calls; their
                        # textual content (if any) is preserved as a plain assistant msg.
                        content = mm.get("content")
                        if content:
                            gap_msgs.append({"role": "assistant", "content": content})
                        continue
                    if role == "tool":
                        # Reframe the tool output as a user message so the model can read it.
                        content = mm.get("content", "")
                        gap_msgs.append({"role": "user", "content": f"[tool result]\n{content}"})
                        continue
                    gap_msgs.append(mm)
                gap_msgs.append({"role": "user", "content": "Based on the retrieved documents above and the original question, identify specific information gaps that are not covered and need to be addressed. Return a concise numbered list of missing pieces of information."})
                try:
                    gap_r = await llm.chat.completions.create(
                        model=llm_cfg["llm_model"],
                        messages=gap_msgs,
                        temperature=llm_cfg.get("temperature", 0),
                        max_completion_tokens=1024,
                    )
                    out = gap_r.choices[0].message.content or "No gaps identified."
                except (oai.BadRequestError, oai.RateLimitError, oai.APIStatusError) as e:
                    print(f"  [find_gaps] LLM error: {e}")
                    out = json.dumps({"error": str(e)})
                print(f"  [find_gaps] {out[:200]}")
            else:
                out = json.dumps({"error": f"Unknown tool: {t.function.name}"})
            return t, out, False
        # Enforce: initial_search must be alone
        if "initial_search" in call_names and len(call_names) > 1:
            print(f"  [warn] initial_search mixed with other calls; returning error for non-initial_search calls")
            for t in m.tool_calls:
                if t.function.name != "initial_search":
                    msgs.append({"role": "tool", "tool_call_id": t.id,
                                 "content": json.dumps({"error": "initial_search must be the only tool call in the first batch; re-issue this call separately."})})
            m_tool_calls = [t for t in m.tool_calls if t.function.name == "initial_search"]
        else:
            m_tool_calls = m.tool_calls
        print(f"  Executing {len(m_tool_calls)} tool calls in parallel...")
        results = await asyncio.gather(*(_exec(t) for t in m_tool_calls))
        pruned = False
        for t, out, is_prune in results:
            if is_prune:
                msgs.clear()
                msgs.append(initial_msg)
                msgs.append({"role": "assistant", "content": "I'll prune the context to focus on the most relevant documents."})
                msgs.append({"role": "user", "content": out})
                try:
                    prune_args = json.loads(t.function.arguments)
                    print(f"  [prune] Kept {len(prune_args['docids'])} docs, context reset")
                except (json.JSONDecodeError, TypeError, KeyError):
                    print(f"  [prune] context reset")
                pruned = True
                break
        if pruned:
            continue
        for t, out, _ in results:
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": out})
            try:
                a = json.loads(t.function.arguments)
                vals = list(a.values())
                print(f"  [{t.function.name}] {vals[0][:80] if vals and isinstance(vals[0], str) else '...'}")
            except (json.JSONDecodeError, TypeError):
                print(f"  [{t.function.name}] (malformed args)")
        non_prune_rounds += 1
        token_est = count_tokens(msgs)
        # If tool results pushed us over the limit, auto-prune immediately
        if token_est > CONTEXT_LIMIT * 0.8 and doc_cache:
            all_ids = list(doc_cache.keys())
            if len(all_ids) > PRUNE_K:
                doc_texts = [doc_cache[did] for did in all_ids]
                ranked_texts = await rerank(query, doc_texts, PRUNE_K)
                text_id_map = {id(doc_texts[i]): all_ids[i] for i in range(len(all_ids))}
                keep_ids = [text_id_map[id(t)] for t in ranked_texts if id(t) in text_id_map]
            else:
                keep_ids = all_ids
            parts = [f'<doc id="{did}">\n{doc_cache[did]}\n</doc>' for did in keep_ids]
            pruned_ctx = "Pruned context (only these documents remain):\n\n" + "\n\n".join(parts)
            msgs.clear()
            msgs.append(initial_msg)
            msgs.append({"role": "assistant", "content": "I'll prune the context to focus on the most relevant documents."})
            msgs.append({"role": "user", "content": pruned_ctx})
            token_est_new = count_tokens(msgs)
            tc["prune"] = tc.get("prune", 0) + 1
            print(f"  [auto-prune] Token overflow after tool results ({token_est} tokens), kept {len(keep_ids)} docs, reset to {token_est_new} tokens")
            token_est = token_est_new
        msgs.append({"role": "user", "content": f"Token usage: {token_est} / {CONTEXT_LIMIT}. Non-prune tool calls since start: {non_prune_rounds}"})
        print(f"  Token usage: {token_est} / {CONTEXT_LIMIT}. Non-prune rounds: {non_prune_rounds}")
    elapsed = round(time.perf_counter() - t0, 2)
    return {"question_id": qid, "query": query, "answer": "", "ground_truth": q_obj.get("answer",""),
            "model": llm_cfg["llm_model"], "rounds": 50, "elapsed_seconds": elapsed, "tool_calls": tc}

async def main():
    questions = json.loads(Path(cfg["paths"]["questions_path"]).read_text())
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
    use_rbac_auth = cosmos_cfg.get("use_rbac_auth", False)
    credential = AsyncAzureCliCredential() if use_rbac_auth else None
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=credential or cosmos_cfg["key"])
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    containers = {s["id"]: db.get_container_client(s["container_name"]) for s in sources}

    try:
        results = []
        for q in tqdm(questions):
            results.append(await process_question(q, containers))

        out = Path(cfg["paths"]["output_root"]) / "standard" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {len(results)} results to {out}")
    finally:
        await cosmos.close()
        if credential is not None:
            await credential.close()
        await llm.close()
        await embed_client.close()
        if USE_RANKER:
            await _r_http.aclose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_dynamic.yaml")
    parser.add_argument("--max-questions", type=int, default=None, help="Only answer the first N questions")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    llm_cfg, embed_cfg, cosmos_cfg = cfg["llm"], cfg["embedding"], cfg["cosmos"]
    sources = cosmos_cfg["sources"]
    _source_cfg = {s["id"]: s["retrieval"] for s in sources}
    _source_embed = {s["id"]: s["embedding_field"] for s in sources}
    _source_ft = {s["id"]: s["retrieval"]["fulltext_fields"] for s in sources}
    _all_embed = set(_source_embed.values())
    MAX_RETRIES, RERANK_MUL = int(llm_cfg["max_retries"]), cfg["ranker"]["rerank_multiplier"]
    PRUNE_K = cfg.get("prune_k", 20)
    CONTEXT_LIMIT = llm_cfg.get("context_limit", 270000)
    USE_HYDE = cfg["hyde"]

    # Clients
    tp = get_bearer_token_provider(AzureCliCredential(), llm_cfg["token_scope"]) if llm_cfg["use_rbac_auth"] else None
    llm = AsyncAzureOpenAI(api_version=llm_cfg["api_version"], azure_endpoint=llm_cfg["llm_endpoint"],
        **({"azure_ad_token_provider": tp} if tp else {"api_key": llm_cfg["llm_api_key"]}))
    embed_tp = get_bearer_token_provider(AzureCliCredential(), embed_cfg["token_scope"]) if embed_cfg.get("use_rbac_auth") else None
    if embed_cfg.get("use_rbac_auth") or embed_cfg.get("embed_api_key"):
        embed_client = AsyncAzureOpenAI(api_version=embed_cfg["api_version"], azure_endpoint=embed_cfg["embed_endpoint"],
            **({"azure_ad_token_provider": embed_tp} if embed_tp else {"api_key": embed_cfg["embed_api_key"]}))
    else:
        embed_client = oai.AsyncOpenAI(base_url=embed_cfg["embed_endpoint"], api_key="ollama")

    # Ranker
    rcfg = cfg["ranker"]
    USE_RANKER = rcfg["use_ranker"]
    if USE_RANKER:
        _r_url = build_ranker_url(rcfg)
        _r_bs, _r_mr = rcfg["batch_size"], rcfg["max_retries"]
        token_scope = str(rcfg.get("token_scope") or DEFAULT_MANAGEMENT_SCOPE).strip()
        _r_tok = _get_cli_token(rcfg, token_scope)
        _r_http = httpx.AsyncClient(timeout=120)

    from prompts import DEFAULT_QUERY_TEMPLATE
    QUERY_TEMPLATE = DEFAULT_QUERY_TEMPLATE.replace("{prune_k}", str(PRUNE_K))

    asyncio.run(main())
