#!/usr/bin/env python
"""Self-contained Decomposed RAG with fulltext + vector search (concise version)."""

import argparse
import json
import os
import re
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import openai
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential, AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
import numpy as np

load_dotenv()

# Load config from yaml
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================

@dataclass
class Question:
    question_id: str
    question_text: str
    group: str | None = None
    ground_truth: str | None = None

@dataclass
class RetrievedChunk:
    chunk_id: int | str
    text: str
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SubQuestionResult:
    sub_question: str
    retrieved_chunks: list[dict]
    answer: str

@dataclass
class RoundResult:
    round_num: int
    preliminary_answer_before: str
    sub_question_results: list[SubQuestionResult]
    regenerated_answer: str | None

# =============================================================================
# PROMPTS
# =============================================================================

PRELIMINARY_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Context Documents:
{context}

Question: {question}

Provide your answer in the following format:

ANSWER FROM CONTEXT:
[Your answer based strictly on the provided context]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context, or write "None" if the context fully answers the question]

Response:"""

SUBQUESTION_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Provide a COMPREHENSIVE and DETAILED answer - include all relevant information from the context
7. Extract and include specific values, numbers, specifications, and technical details when available
8. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Context Documents:
{context}

Question: {question}

Provide a VERBOSE and COMPREHENSIVE answer that includes ALL relevant information from the context.
Include specific details, values, and technical specifications where available.

ANSWER FROM CONTEXT:
[Your detailed answer based strictly on the provided context - be thorough and include all relevant details]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context, or write "None" if the context fully answers the question]

Response:"""

REGENERATE_PROMPT = """You are a helpful assistant that synthesizes information to provide a comprehensive answer.

You have:
1. An original question
2. A previous preliminary answer (which may have gaps)
3. Additional information from follow-up sub-questions and their answers

Your task is to generate an UPDATED and MORE COMPLETE answer by incorporating the new information from the sub-questions.

IMPORTANT RULES:
1. ONLY use information from the previous answer and the sub-question answers provided
2. DO NOT make assumptions or add external knowledge
3. Integrate the new information smoothly into a coherent answer
4. If gaps still remain, clearly identify them
5. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Original Question: {question}

Previous Preliminary Answer:
{previous_answer}

Additional Information from Sub-questions:
{sub_qa_context}

Provide your updated answer in the following format:

ANSWER FROM CONTEXT:
[Your updated answer incorporating all available information]

INFORMATION GAPS:
[List any aspects of the question that still cannot be answered, or write "None" if the question is now fully answered]

Response:"""

GAP_DECOMPOSE_PROMPT = """You are a helpful assistant that identifies what additional information is needed to fully answer a question.

Given:
1. An original question
2. A preliminary answer based on initial context (which may be incomplete)
3. The information gaps identified in that answer

Generate sub-questions that will help fill these gaps and provide a complete answer.

Guidelines:
- Focus on the INFORMATION GAPS identified in the preliminary answer
- Each sub-question should be SIMPLE and cover only ONE specific aspect
- Each sub-question should target a single missing piece of information
- Do NOT combine multiple aspects into one sub-question
- Sub-questions should be self-contained and answerable independently
- Keep sub-questions short and focused
- Maximum {max_sub_questions} sub-questions

Original Question: {question}

Preliminary Answer:
{preliminary_answer}

Return your response as a JSON array of strings. Return [] if no additional information is needed.
Example: ["What is the maximum temperature?", "What is the minimum temperature?"]

Sub-questions to fill gaps:"""

SYNTHESIS_PROMPT = """You are a helpful assistant that synthesizes information to answer questions comprehensively.

Original Question: {original_question}

Preliminary Answer (from initial retrieval):
{preliminary_answer}

Sub-questions and their answers:
{sub_qa_pairs}

Based on the above information, provide a comprehensive answer to the original question.
Synthesize the information coherently, avoid repetition, and ensure the answer directly addresses the original question.

Prioritize information from the sub-question answers that fill gaps in the preliminary answer.

At the end, add a summary that directly answers the question in a concise way. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Format your response as:
[Your comprehensive answer here]

SUMMARY: [Direct answer to the question]

Final Answer:"""

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    def __init__(self):
        llm_cfg = CONFIG["llm"]
        self._use_rbac_auth = bool(llm_cfg["use_rbac_auth"])
        self._api_key = llm_cfg["azure_openai_key"]
        self._token_provider = None
        if self._use_rbac_auth:
            token_scope = llm_cfg.get("token_scope")
            if not token_scope or not str(token_scope).strip():
                token_scope = "https://cognitiveservices.azure.com/.default"
            self._token_provider = get_bearer_token_provider(AzureCliCredential(), token_scope)
        self._llm_client = None
        self._embed_client = None
        self._cfg = llm_cfg
        self._embed_dimensions = int(llm_cfg.get("embed_dimensions") or 0)

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        if self._embed_dimensions <= 0:
            return [float(x) for x in embedding]
        values = [float(x) for x in embedding]
        if len(values) > self._embed_dimensions:
            return values[:self._embed_dimensions]
        if len(values) < self._embed_dimensions:
            return values + [0.0] * (self._embed_dimensions - len(values))
        return values
    
    @property
    def llm_client(self) -> AzureOpenAI:
        if not self._llm_client:
            client_kwargs = {
                "api_version": self._cfg["api_version"],
                "azure_endpoint": self._cfg["llm_endpoint"],
            }
            if self._use_rbac_auth:
                client_kwargs["azure_ad_token_provider"] = self._token_provider
            else:
                if not self._api_key or not str(self._api_key).strip():
                    raise ValueError("llm.azure_openai_key must be set when llm.use_rbac_auth is false")
                client_kwargs["api_key"] = self._api_key
            self._llm_client = AzureOpenAI(**client_kwargs)
        return self._llm_client
    
    @property
    def embed_client(self) -> AzureOpenAI:
        if not self._embed_client:
            client_kwargs = {
                "api_version": self._cfg["api_version"],
                "azure_endpoint": self._cfg["embed_endpoint"],
            }
            if self._use_rbac_auth:
                client_kwargs["azure_ad_token_provider"] = self._token_provider
            else:
                if not self._api_key or not str(self._api_key).strip():
                    raise ValueError("llm.azure_openai_key must be set when llm.use_rbac_auth is false")
                client_kwargs["api_key"] = self._api_key
            self._embed_client = AzureOpenAI(**client_kwargs)
        return self._embed_client
    
    def complete(self, prompt: str, retries: int = None) -> str:
        if retries is None:
            retries = self._cfg["max_retries"]
        for attempt in range(retries):
            try:
                result = self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._cfg["llm_model"],
                    temperature=self._cfg["temperature"],
                    max_completion_tokens=self._cfg["max_completion_tokens"],
                )
                return result.choices[0].message.content
            except openai.RateLimitError:
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                print(f"Rate limited, retry in {wait}s ({attempt + 1}/{retries})")
                time.sleep(wait)
            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
                wait = min(5.0 * (2 ** attempt), 5.0 * (2 ** 8))
                print(f"API error ({type(e).__name__}), retry in {wait}s ({attempt + 1}/{retries})")
                time.sleep(wait)
        raise Exception("Max retries exceeded")
    
    def embed(self, text: str) -> list[float]:
        embed_endpoint = str(self._cfg.get("embed_endpoint", "")).strip()
        if embed_endpoint.endswith("/api/embeddings"):
            payload = json.dumps({"model": self._cfg["embed_model"], "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                embed_endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                embedding = parsed.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError("Invalid Ollama embedding response: missing 'embedding' list")
                return self._normalize_embedding(embedding)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")
                except Exception:
                    pass
                raise RuntimeError(f"Embedding endpoint HTTP {e.code}: {detail[:300]}") from e

        result = self.embed_client.embeddings.create(input=[text], model=self._cfg["embed_model"])
        return self._normalize_embedding(result.data[0].embedding)

# =============================================================================
# COSMOS DB RETRIEVER
# =============================================================================

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", CONFIG["cosmos"]["uri"])
COSMOS_KEY = os.getenv("COSMOS_KEY", CONFIG["cosmos"]["key"])
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", CONFIG["cosmos"]["database_name"])
CONTAINER_STRUCTURED = os.getenv("COSMOS_STRUCTURED_CONTAINER_NAME", CONFIG["cosmos"]["structured_container"])
CONTAINER_UNSTRUCTURED = os.getenv("COSMOS_UNSTRUCTURED_CONTAINER_NAME", CONFIG["cosmos"]["unstructured_container"])
COSMOS_STRUCTURED_PARTITION_KEY_PATH = os.getenv(
    "COSMOS_STRUCTURED_PARTITION_KEY_PATH",
    CONFIG["cosmos"]["structured_partition_key_path"],
)
COSMOS_UNSTRUCTURED_PARTITION_KEY_PATH = os.getenv(
    "COSMOS_UNSTRUCTURED_PARTITION_KEY_PATH",
    CONFIG["cosmos"]["unstructured_partition_key_path"],
)

# Comprehensive BM25 stopwords list
STOPWORDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "make", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}

def greedy_log_det_select(vectors: np.ndarray, query_vec: np.ndarray, k: int,
                          eta: float = 0.0, rescale_power: float = 0.0) -> list[int]:
    """Greedily select k indices maximizing log-det(Gram) for diversity."""
    V = vectors.copy()
    if rescale_power > 0:
        sims = V @ query_vec
        for i in range(len(V)):
            V[i] *= (sims[i] ** rescale_power) if sims[i] > 0 else 0
    n = len(V)
    if k >= n:
        return list(range(n))
    chosen, sel_vecs = [], []
    for _ in range(k):
        best_i, best_ld = -1, -np.inf
        for i in range(n):
            if i in chosen:
                continue
            test = np.vstack(sel_vecs + [V[i]]) if sel_vecs else V[i:i+1]
            gram = test @ test.T
            if eta > 0:
                gram += eta * np.eye(len(gram))
            _, ld = np.linalg.slogdet(gram)
            if ld > best_ld:
                best_ld, best_i = ld, i
        chosen.append(best_i)
        sel_vecs.append(V[best_i])
    return chosen


class CombinedRetriever:
    def __init__(self, k_fulltext: int, k_structured: int, k_unstructured: int,
                 k_diverse: int = 0, eta: float = 0.0, rescale_power: float = 0.0):
        self.k_fulltext = k_fulltext
        self.k_structured = k_structured
        self.k_unstructured = k_unstructured
        self.k_diverse = k_diverse
        self.eta = eta
        self.rescale_power = rescale_power
        self._cosmos = None
        self._db = None
        self._structured = None
        self._unstructured = None
        self._llm = None
        self._expected_vector_dim = int(CONFIG.get("llm", {}).get("embed_dimensions") or 0)
        self._structured_partition_key_path = str(COSMOS_STRUCTURED_PARTITION_KEY_PATH or "").strip()
        self._unstructured_partition_key_path = str(COSMOS_UNSTRUCTURED_PARTITION_KEY_PATH or "").strip()

    def _partition_key_expr(self, partition_key_path: str) -> str:
        segments = [s for s in partition_key_path.split("/") if s]
        if not segments:
            raise ValueError("Cosmos partition key path is not configured")
        return "c." + ".".join(segments)
    
    def initialize(self):
        #self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=DefaultAzureCredential())
        self._cosmos = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY, connection_mode="Direct")
        self._db = self._cosmos.get_database_client(DATABASE_NAME)
        self._structured = self._db.get_container_client(CONTAINER_STRUCTURED)
        self._unstructured = self._db.get_container_client(CONTAINER_UNSTRUCTURED)
        self._llm = LLMClient()
    
    def _fulltext_search(self, query: str, top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        terms = [t for t in re.findall(r'\w+', query) if t.lower() not in STOPWORDS and len(t) > 1]
        if not terms:
            return []
        
        # Chunk terms (max 5 per FullTextScore)
        chunks = [terms[i:i+5] for i in range(0, len(terms), 5)]
        if len(chunks) == 1:
            args = ', '.join(f'"{t}"' for t in chunks[0])
            order = f"ORDER BY RANK FullTextScore(c.designation, {args})"
        else:
            scores = [f"FullTextScore(c.designation, {', '.join(f'\"{t}\"' for t in c)})" for c in chunks]
            order = f"ORDER BY RANK RRF({', '.join(scores)})"
        
        try:
            return list(self._structured.query_items(
                query=f"SELECT TOP {top_k} * FROM c {order}",
                parameters=[],
                enable_cross_partition_query=True
            ))
        except Exception as e:
            print(f"Fulltext error: {e}")
            return []
    
    def _vector_search(self, container, partition_key_path: str, query_emb: list[float], top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        adjusted_emb = [float(x) for x in query_emb]
        if self._expected_vector_dim > 0:
            if len(adjusted_emb) > self._expected_vector_dim:
                adjusted_emb = adjusted_emb[:self._expected_vector_dim]
            elif len(adjusted_emb) < self._expected_vector_dim:
                adjusted_emb = adjusted_emb + [0.0] * (self._expected_vector_dim - len(adjusted_emb))
        pk_expr = self._partition_key_expr(partition_key_path)
        results = list(container.query_items(
            query=f"SELECT TOP @k c.id, {pk_expr} AS pkv, VectorDistance(c.e, @emb) AS score FROM c WHERE IS_DEFINED(c.e) ORDER BY VectorDistance(c.e, @emb)",
            parameters=[{"name": "@k", "value": top_k}, {"name": "@emb", "value": adjusted_emb}],
            enable_cross_partition_query=True
        ))
        docs = []
        for r in results:
            partition_key_value = r.get("pkv")
            if partition_key_value is None:
                continue
            doc = container.read_item(item=r["id"], partition_key=partition_key_value)
            doc["_score"] = r.get("score")
            docs.append(doc)
        return docs
    
    def _format_doc(self, doc: dict, source: str) -> RetrievedChunk:
        embedding = doc.get('e') if isinstance(doc.get('e'), list) else doc.get('embedding')
        exclude = {'_rid', '_self', '_etag', '_attachments', '_ts', 'embedding', 'e', '_score'}
        parts = []
        for k, v in doc.items():
            if k not in exclude and v:
                parts.append(f"{k.replace('_', ' ').title()}: {v if not isinstance(v, (list, dict)) else str(v)}")
        return RetrievedChunk(
            chunk_id=doc.get('id', ''),
            text="\n".join(parts),
            similarity=1 - doc.get('_score', 0) if '_score' in doc else None,
            metadata={'_data_source': source, 'embedding': embedding}
        )
    
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        chunks, seen = [], set()
        
        # Fulltext
        for doc in self._fulltext_search(query, self.k_fulltext):
            if doc.get('id') not in seen:
                seen.add(doc['id'])
                chunks.append(self._format_doc(doc, 'fulltext'))
        
        # Vector searches
        emb = self._llm.embed(query)
        for doc in self._vector_search(self._structured, self._structured_partition_key_path, emb, self.k_structured):
            if doc.get('id') not in seen:
                seen.add(doc['id'])
                chunks.append(self._format_doc(doc, 'structured_vector'))
        for doc in self._vector_search(self._unstructured, self._unstructured_partition_key_path, emb, self.k_unstructured):
            if doc.get('id') not in seen:
                seen.add(doc['id'])
                chunks.append(self._format_doc(doc, 'unstructured_vector'))
        
        # Diversity selection via greedy log-det maximization
        if self.k_diverse > 0 and len(chunks) > self.k_diverse:
            for c in chunks:
                if c.metadata.get('embedding') is None:
                    c.metadata['embedding'] = self._llm.embed(c.text)
            vectors = np.array([c.metadata['embedding'] for c in chunks], dtype=np.float32)
            query_vec = np.array(emb, dtype=np.float32)
            selected = greedy_log_det_select(vectors, query_vec, self.k_diverse, self.eta, self.rescale_power)
            chunks = [chunks[i] for i in selected]
        
        return chunks

# =============================================================================
# DECOMPOSED RAG PIPELINE
# =============================================================================

class DecomposedRAGPipeline:
    def __init__(self, retriever: CombinedRetriever, llm: LLMClient, max_sub_q: int = 5, num_rounds: int = 2):
        self.retriever = retriever
        self.llm = llm
        self.max_sub_q = max_sub_q
        self.num_rounds = num_rounds
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    
    def _get_subquestions(self, question: str, answer: str) -> list[str]:
        resp = self.llm.complete(GAP_DECOMPOSE_PROMPT.format(
            question=question, preliminary_answer=answer, max_sub_questions=self.max_sub_q
        ))
        try:
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                subs = json.loads(match.group())
                if isinstance(subs, list):
                    return [s for s in subs if isinstance(s, str)][:self.max_sub_q]
        except:
            pass
        return []
    
    def _answer_subquestion(self, sub_q: str) -> SubQuestionResult:
        chunks = self.retriever.retrieve(sub_q)
        context = self._format_context(chunks)
        answer = self.llm.complete(SUBQUESTION_PROMPT.format(context=context, question=sub_q))
        return SubQuestionResult(
            sub_question=sub_q,
            retrieved_chunks=[{"chunk_id": c.chunk_id, "content": c.text, "metadata": {k: v for k, v in c.metadata.items() if k != 'embedding'}} for c in chunks],
            answer=answer
        )
    
    def run(self, question: str) -> dict:
        # Initial retrieval
        initial_chunks = self.retriever.retrieve(question)
        initial_context = self._format_context(initial_chunks)
        preliminary = self.llm.complete(PRELIMINARY_PROMPT.format(context=initial_context, question=question))
        
        rounds, all_subs = [], []
        current = preliminary
        
        for rnd in range(1, self.num_rounds + 1):
            sub_qs = self._get_subquestions(question, current)
            if not sub_qs:
                break
            
            # Process sub-questions in parallel
            with ThreadPoolExecutor(max_workers=len(sub_qs)) as ex:
                sub_results = list(ex.map(self._answer_subquestion, sub_qs))
            all_subs.extend(sub_results)
            
            if rnd < self.num_rounds:
                # Regenerate
                sub_ctx = "\n\n".join(f"Q: {s.sub_question}\nA: {s.answer}" for s in all_subs)
                regen = self.llm.complete(REGENERATE_PROMPT.format(
                    question=question, previous_answer=current, sub_qa_context=sub_ctx
                ))
                rounds.append(RoundResult(rnd, current, sub_results, regen))
                current = regen
            else:
                rounds.append(RoundResult(rnd, current, sub_results, None))
        
        # Synthesize
        sub_pairs = "\n\n".join(f"Q{i+1}: {s.sub_question}\nA{i+1}: {s.answer}" for i, s in enumerate(all_subs))
        final = self.llm.complete(SYNTHESIS_PROMPT.format(
            original_question=question, preliminary_answer=current, sub_qa_pairs=sub_pairs or "None"
        ))
        
        return {
            "initial_chunks": [{"id": c.chunk_id, "src": c.metadata.get('_data_source'), "content": c.text} for c in initial_chunks],
            "initial_answer": preliminary,
            "rounds": [{
                "round": r.round_num,
                "sub_questions": [{"q": s.sub_question, "a": s.answer, "chunks": s.retrieved_chunks} for s in r.sub_question_results],
                "regenerated": r.regenerated_answer
            } for r in rounds],
            "final_answer": final
        }

# =============================================================================
# MAIN
# =============================================================================

def load_questions(path: Path) -> list[Question]:
    questions = []
    for f in path.glob("*.json"):
        if f.stem.startswith("_") or f.stem.endswith("_test_query"):
            continue
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        questions.extend(Question(q["question_id"], q["question_text"], f.stem, q.get("answer")) for q in data)
    return questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Override config yaml path")
    parser.add_argument("--k-fulltext", type=int, default=CONFIG["retrieval"]["k_fulltext"])
    parser.add_argument("--k-structured", type=int, default=CONFIG["retrieval"]["k_structured"])
    parser.add_argument("--k-unstructured", type=int, default=CONFIG["retrieval"]["k_unstructured"])
    parser.add_argument("--k-diverse", type=int, default=CONFIG["retrieval"]["k_diverse"], help="Diverse chunks to select via log-det (0=disabled)")
    parser.add_argument("--eta", type=float, default=CONFIG["retrieval"]["eta"], help="Gram matrix regularization")
    parser.add_argument("--rescale-power", type=float, default=CONFIG["retrieval"]["rescale_power"], help="Query-similarity rescale power")
    parser.add_argument("--max-sub-questions", type=int, default=CONFIG["pipeline"]["max_sub_questions"])
    parser.add_argument("--rounds", type=int, default=CONFIG["pipeline"]["rounds"])
    parser.add_argument("--max-questions", type=int, default=CONFIG["execution"]["max_questions"])
    parser.add_argument("--max-workers", type=int, default=CONFIG["execution"]["max_workers"])
    parser.add_argument("--questions-path", type=Path, default=Path(CONFIG["paths"]["questions_path"]))
    parser.add_argument("--output-root", type=Path, default=Path(CONFIG["paths"]["output_root"]))
    args = parser.parse_args()
    
    total_k = args.k_fulltext + args.k_structured + args.k_unstructured
    print(f"Decomposed RAG: fulltext={args.k_fulltext}, structured={args.k_structured}, unstructured={args.k_unstructured}, diverse={args.k_diverse}")
    
    retriever = CombinedRetriever(args.k_fulltext, args.k_structured, args.k_unstructured,
                                  args.k_diverse, args.eta, args.rescale_power)
    retriever.initialize()
    llm = LLMClient()
    pipeline = DecomposedRAGPipeline(retriever, llm, args.max_sub_questions, args.rounds)
    
    questions = load_questions(args.questions_path)
    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Processing {len(questions)} questions")
    
    div_suffix = f"_div{args.k_diverse}" if args.k_diverse > 0 else ""
    output_path = args.output_root / f"k{total_k}_ft{args.k_fulltext}_str{args.k_structured}_uns{args.k_unstructured}{div_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results, lock = [], threading.Lock()
    
    def process(q):
        result = pipeline.run(q.question_text)
        result["question_id"] = q.question_id
        result["question_text"] = q.question_text
        result["group"] = q.group
        result["ground_truth"] = q.ground_truth
        (output_path / "intermediate" / q.group).mkdir(parents=True, exist_ok=True)
        (output_path / "intermediate" / q.group / f"{q.question_id}.json").write_text(json.dumps(result, indent=2))
        return result
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process, q): q for q in questions}
        for f in tqdm(as_completed(futures), total=len(questions)):
            try:
                with lock:
                    results.append(f.result())
            except Exception as e:
                print(f"Error: {futures[f].question_id}: {e}")
    
    # Save final results - group by question group
    grouped = {}
    for r in results:
        group = r.get("group", "default")
        if group not in grouped:
            grouped[group] = []
        grouped[group].append({
            "question_id": r["question_id"],
            "question_text": r["question_text"],
            "answer": r["final_answer"],
            "ground_truth": r.get("ground_truth")
        })
    (output_path / "questions_with_answers.json").write_text(json.dumps(grouped, indent=2))
    print(f"Done! Results: {output_path}")

if __name__ == "__main__":
    main()
