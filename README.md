# DiverseRAG

DiverseRAG is a two-stage Azure Cosmos DB + Azure OpenAI pipeline:

1. **Ingestion (`cosmos_db_upload.py`)**
   - Reads structured and/or unstructured JSON documents from local folders.
   - Builds embeddings and stores them in field `e`.
   - Upserts documents into Cosmos DB containers with vector + full-text indexing support.

2. **Retrieval + Answering (`rag_divdet.py`)**
   - Runs decomposed RAG using full-text retrieval, vector retrieval, and diversity selection.
   - Generates answers for question files and writes grouped outputs under `out/`.

## What this project does

- Uploads your corpus to Cosmos DB in two optional streams:
  - **Structured** container (e.g., product/taxonomy records)
  - **Unstructured** container (e.g., text/content records)
- Embeds both streams with one configured embedding endpoint/model.
- Answers evaluation questions by combining:
  - Initial retrieval
  - Gap-aware sub-question decomposition
  - Regeneration/synthesis into a final answer

## Prerequisites

- Python 3.10+
- Azure Cosmos DB account + database/containers (or management settings for auto-create)
- Azure OpenAI (or local embedding endpoint if configured)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or use setup helpers:

- PowerShell: `./run.ps1`
- Bash: `source ./run.sh`

## Sequence of actions

### 1) Populate `config.yaml`

Start from `config.yaml.example` and fill required values in `config.yaml`.

At minimum, set:

- `llm.llm_endpoint`
- `llm.embed_endpoint`
- `llm.llm_model`
- `llm.embed_model`
- `llm.azure_openai_key` (if not using RBAC for OpenAI)
- `cosmos.uri`
- `cosmos.key`
- `cosmos.database_name`
- `cosmos.structured_container`
- `cosmos.unstructured_container`
- `cosmos.structured_partition_key_path`
- `cosmos.unstructured_partition_key_path`
- `cosmos.structured_documents_root` and/or `cosmos.unstructured_documents_root`
- `paths.questions_path`
- `paths.output_root`

Optional but recommended for auto-creating missing containers:

- `cosmos.azure_subscription_id`
- `cosmos.cosmos_resource_group`
- `cosmos.cosmos_account_name` (or let script infer from `cosmos.uri`)

### 2) Upload documents to Cosmos DB

Run:

```bash
python cosmos_db_upload.py
```

Notes:

- Upload target(s) are inferred from whether `structured_documents_root` and/or `unstructured_documents_root` are populated.
- If one root is empty, that branch is skipped.
- Override both configured roots with one folder:

```bash
python cosmos_db_upload.py --folder /path/to/documents
```

### 3) Run retrieval and generate answers

Run:

```bash
python rag_divdet.py
```

Typical limited smoke test:

```bash
python rag_divdet.py --max-questions 1
```

Outputs are written to:

- `out/k.../intermediate/...` (per-question intermediate traces)
- `out/k.../questions_with_answers.json` (final grouped answers)

## Useful runtime overrides

`rag_divdet.py` supports CLI overrides for retrieval/pipeline settings, including:

- `--k-fulltext`
- `--k-structured`
- `--k-unstructured`
- `--k-diverse`
- `--eta`
- `--rescale-power`
- `--max-sub-questions`
- `--rounds`
- `--max-questions`
- `--max-workers`
- `--questions-path`
- `--output-root`

## Repository layout

- `cosmos_db_upload.py` — ingestion + embedding + Cosmos upsert
- `rag_divdet.py` — decomposed RAG retrieval/answer pipeline
- `config.yaml.example` — full config template
- `data/` — sample input corpus
- `out/` — generated outputs
