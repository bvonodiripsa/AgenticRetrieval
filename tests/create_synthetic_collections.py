"""Create Synthetic Cosmos DB Collections

This script creates a Cosmos DB database and containers populated from
JSONL sources, as defined in tests/config.test.yaml.

Container names, partition keys, embedding fields, text fields, and JSONL
source paths are all read from the ``cosmos.sources`` list in the config.

Usage:
    python tests/create_synthetic_collections.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
from azure.mgmt.cosmosdb.models import (
    AutoscaleSettings,
    Capability,
    ContainerPartitionKey,
    CreateUpdateOptions,
    DatabaseAccountUpdateParameters,
    ExcludedPath,
    FullTextIndexPath,
    FullTextPath,
    FullTextPolicy,
    IncludedPath,
    IndexingPolicy,
    SqlContainerCreateUpdateParameters,
    SqlContainerResource,
    SqlDatabaseCreateUpdateParameters,
    SqlDatabaseResource,
    VectorEmbedding,
    VectorEmbeddingPolicy,
    VectorIndex,
    VectorIndexType,
)
from openai import AsyncAzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = Path(__file__).parent / "config.test.yaml"

with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

_COSMOS_CFG = CONFIG.get("cosmos", {})
_EMBED_CFG = CONFIG.get("embedding", {})

COSMOS_ENDPOINT: str = str(_COSMOS_CFG.get("uri", "")).strip()
COSMOS_KEY: str = str(_COSMOS_CFG.get("key", "")).strip()
DATABASE_NAME: str = str(_COSMOS_CFG.get("database_name", "divdet")).strip()
USE_RBAC_AUTH: bool = bool(_COSMOS_CFG.get("use_rbac_auth", False))

COSMOS_ACCOUNT_NAME: str = str(_COSMOS_CFG.get("cosmos_account_name", "")).strip()
COSMOS_RESOURCE_GROUP: str = str(_COSMOS_CFG.get("cosmos_resource_group", "")).strip()
AZURE_SUBSCRIPTION_ID: str = str(_COSMOS_CFG.get("azure_subscription_id", "")).strip()

EMBED_ENDPOINT: str = str(_EMBED_CFG.get("embed_endpoint", "")).strip().strip('"')
EMBED_MODEL: str = str(_EMBED_CFG.get("embed_model", "")).strip()
EMBED_DIMENSIONS: int = int(_EMBED_CFG.get("embed_dimensions", 1024))
EMBED_API_VERSION: str = str(_EMBED_CFG.get("api_version", "2024-05-01-preview"))
EMBED_API_KEY: str = str(_EMBED_CFG.get("embed_api_key", "") or "").strip()

# Autoscale max throughput (RU/s) for each container
AUTOSCALE_MAX_THROUGHPUT = 1000

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

SOURCES = _COSMOS_CFG.get("sources", [])


def _load_documents_from_jsonl(source_path: Path) -> list[dict[str, Any]]:
    """Load documents from a JSONL file, preserving original field names."""
    documents: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {source_path} at line {line_num}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                continue
            documents.append(record)
    if not documents:
        raise ValueError(f"No JSON objects found in {source_path}.")
    return documents


def _build_collection_specs() -> list[dict[str, Any]]:
    """Build collection specs from config sources, loading JSONL documents."""
    if not SOURCES:
        raise ValueError("No sources defined in cosmos.sources config.")
    specs: list[dict[str, Any]] = []
    for source in SOURCES:
        documents_root = source.get("documents_root", "")
        if not documents_root:
            raise ValueError(f"Source '{source.get('id', '?')}' has no documents_root.")
        source_path = _REPO_ROOT / documents_root
        if not source_path.exists():
            raise ValueError(f"JSONL source not found: {source_path}")
        docs = _load_documents_from_jsonl(source_path)
        specs.append({
            "container_name": source["container_name"],
            "partition_key_path": source.get("partition_key_path", "/id"),
            "embedding_field": source.get("embedding_field", "e"),
            "embedding_text_fields": source.get("embedding_text_fields", []),
            "source_path": str(source_path),
            "documents": docs,
        })
    return specs

# ---------------------------------------------------------------------------
# Indexing / policy helpers
# ---------------------------------------------------------------------------


def _build_indexing_policy(text_fields: list[str], embedding_field: str = "e") -> dict[str, Any]:
    """Return the indexing policy dict for a collection with the given text fields."""
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"},
            {"path": f"/{embedding_field}/*"},
        ],
        "fullTextIndexes": [{"path": f"/{f}"} for f in text_fields],
        "vectorIndexes": [
            {
                "path": f"/{embedding_field}",
                "type": "diskANN",
                "quantizationByteSize": 192,
                "indexingSearchListSize": 100,
            }
        ],
    }


def _build_full_text_policy(text_fields: list[str]) -> dict[str, Any]:
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": f"/{f}", "language": "en-US"} for f in text_fields],
    }


# ---------------------------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------------------------


def _get_account_name() -> str:
    if COSMOS_ACCOUNT_NAME:
        return COSMOS_ACCOUNT_NAME
    import re
    match = re.match(r"https://([^.]+)\.documents\.azure\.com", COSMOS_ENDPOINT)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot derive account name from endpoint: {COSMOS_ENDPOINT}")


def _ensure_capabilities(credential: SyncDefaultAzureCredential) -> None:
    """Enable NoSQL Vector Search and Full Text Search capabilities if needed."""
    mgmt = CosmosDBManagementClient(credential, AZURE_SUBSCRIPTION_ID)
    account = mgmt.database_accounts.get(COSMOS_RESOURCE_GROUP, _get_account_name())
    existing = {c.name for c in (account.capabilities or [])}
    to_add = []
    for cap in ("EnableNoSQLVectorSearch", "EnableNoSQLFullTextSearch"):
        if cap not in existing:
            print(f"  Enabling capability: {cap}")
            to_add.append(Capability(name=cap))
    if to_add:
        poller = mgmt.database_accounts.begin_update(
            resource_group_name=COSMOS_RESOURCE_GROUP,
            account_name=_get_account_name(),
            update_parameters=DatabaseAccountUpdateParameters(
                capabilities=list(account.capabilities or []) + to_add
            ),
        )
        print("  Waiting for capabilities to propagate…")
        poller.result()
        print("  ✓ Capabilities enabled")
    else:
        print("  ✓ Required capabilities already enabled")


def _create_database(mgmt: CosmosDBManagementClient) -> None:
    account_name = _get_account_name()
    try:
        params = SqlDatabaseCreateUpdateParameters(resource=SqlDatabaseResource(id=DATABASE_NAME))
        mgmt.sql_resources.begin_create_update_sql_database(
            resource_group_name=COSMOS_RESOURCE_GROUP,
            account_name=account_name,
            database_name=DATABASE_NAME,
            create_update_sql_database_parameters=params,
        ).result()
        print(f"  ✓ Database '{DATABASE_NAME}' ready")
    except Exception as exc:
        if "Conflict" in str(exc) or "already exists" in str(exc).lower():
            print(f"  ✓ Database '{DATABASE_NAME}' already exists")
        else:
            raise


def _container_exists_mgmt(mgmt: CosmosDBManagementClient, container_name: str) -> bool:
    try:
        mgmt.sql_resources.get_sql_container(
            resource_group_name=COSMOS_RESOURCE_GROUP,
            account_name=_get_account_name(),
            database_name=DATABASE_NAME,
            container_name=container_name,
        )
        return True
    except HttpResponseError as exc:
        if getattr(exc, "status_code", None) == 404:
            return False
        raise


def _is_serverless(mgmt: CosmosDBManagementClient) -> bool:
    """Return True if the Cosmos DB account is serverless."""
    account = mgmt.database_accounts.get(COSMOS_RESOURCE_GROUP, _get_account_name())
    capabilities = {c.name for c in (account.capabilities or [])}
    return "EnableServerless" in capabilities


def _create_container(
    mgmt: CosmosDBManagementClient,
    container_name: str,
    partition_key: str,
    text_fields: list[str],
    embedding_field: str = "e",
    serverless: bool = False,
) -> None:
    account_name = _get_account_name()
    if _container_exists_mgmt(mgmt, container_name):
        print(f"  ✓ Container '{container_name}' already exists — skipping creation")
        return

    ip_cfg = _build_indexing_policy(text_fields, embedding_field)
    ftp_cfg = _build_full_text_policy(text_fields)

    indexing_policy = IndexingPolicy(
        indexing_mode=ip_cfg["indexingMode"],
        automatic=ip_cfg["automatic"],
        included_paths=[IncludedPath(path=p["path"]) for p in ip_cfg["includedPaths"]],
        excluded_paths=[ExcludedPath(path=p["path"]) for p in ip_cfg["excludedPaths"]],
        vector_indexes=[
            VectorIndex(path=v["path"], type=VectorIndexType(v["type"]))
            for v in ip_cfg["vectorIndexes"]
        ],
        full_text_indexes=[
            FullTextIndexPath(path=f["path"]) for f in ip_cfg["fullTextIndexes"]
        ],
    )

    vector_policy = VectorEmbeddingPolicy(
        vector_embeddings=[
            VectorEmbedding(
                path=f"/{embedding_field}",
                data_type="float32",
                dimensions=EMBED_DIMENSIONS,
                distance_function="cosine",
            )
        ]
    )

    full_text_policy = FullTextPolicy(
        default_language=ftp_cfg["defaultLanguage"],
        full_text_paths=[
            FullTextPath(path=p["path"], language=p["language"])
            for p in ftp_cfg["fullTextPaths"]
        ],
    )

    resource = SqlContainerResource(
        id=container_name,
        partition_key=ContainerPartitionKey(paths=[partition_key], kind="Hash"),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_policy,
        full_text_policy=full_text_policy,
    )
    if serverless:
        params = SqlContainerCreateUpdateParameters(
            resource=resource,
            options=CreateUpdateOptions(),
        )
    else:
        params = SqlContainerCreateUpdateParameters(
            resource=resource,
            options=CreateUpdateOptions(
                autoscale_settings=AutoscaleSettings(max_throughput=AUTOSCALE_MAX_THROUGHPUT)
            ),
        )

    max_retries = 5
    retry_delay = 30
    for attempt in range(max_retries):
        try:
            # Safety guard: re-check on each attempt in case a concurrent process created it.
            if _container_exists_mgmt(mgmt, container_name):
                print(f"  ✓ Container '{container_name}' already exists — skipping creation")
                return
            mgmt.sql_resources.begin_create_update_sql_container(
                resource_group_name=COSMOS_RESOURCE_GROUP,
                account_name=account_name,
                database_name=DATABASE_NAME,
                container_name=container_name,
                create_update_sql_container_parameters=params,
            ).result()
            print(f"  ✓ Container '{container_name}' created")
            return
        except Exception as exc:
            err = str(exc)
            if "Conflict" in err or "already exists" in err.lower():
                print(f"  ✓ Container '{container_name}' already exists")
                return
            if "capability has not been enabled" in err.lower() and attempt < max_retries - 1:
                print(f"  ⏳ Waiting for capabilities to propagate (attempt {attempt + 1}/{max_retries})…")
                time.sleep(retry_delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _build_embedding_text(doc: dict[str, Any], text_fields: list[str]) -> str:
    """Concatenate text fields into a single string for embedding generation."""
    parts: list[str] = []
    for field in text_fields:
        value = doc.get(field)
        if value:
            label = field.replace("_", " ").title()
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def _get_embed_client() -> AsyncAzureOpenAI:
    endpoint = EMBED_ENDPOINT.rstrip("/")
    azure_endpoint = endpoint
    if "/openai/deployments/" in azure_endpoint:
        azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]
    if not EMBED_API_KEY:
        raise ValueError(
            "embedding.embed_api_key must be set in config for embedding generation"
        )
    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=EMBED_API_KEY,
        api_version=EMBED_API_VERSION,
    )


async def _generate_embeddings(
    client: AsyncAzureOpenAI, texts: list[str]
) -> list[list[float]]:
    response = await client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        dimensions=EMBED_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("Create Synthetic Cosmos DB Collections")
    print("=" * 60)

    # Validate required config
    if not COSMOS_ENDPOINT:
        print("❌ cosmos.uri is not set in config")
        sys.exit(1)
    if not USE_RBAC_AUTH and not COSMOS_KEY:
        print("❌ cosmos.key is not set (or set cosmos.use_rbac_auth: true)")
        sys.exit(1)
    if not EMBED_ENDPOINT:
        print("❌ embedding.embed_endpoint is not set in config")
        sys.exit(1)
    if not EMBED_MODEL:
        print("❌ embedding.embed_model is not set in config")
        sys.exit(1)
    if not EMBED_API_KEY:
        print("❌ embedding.embed_api_key is not set in config")
        sys.exit(1)

    try:
        collection_specs = _build_collection_specs()
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    print("\n📚 Using JSONL sources:")
    for spec in collection_specs:
        print(f"  - {spec['container_name']}: {spec['source_path']} ({len(spec['documents'])} docs)")

    # -----------------------------------------------------------------------
    # Step 1: Create database and containers via management plane
    # -----------------------------------------------------------------------
    print("\n📦 Setting up database and containers…")

    if AZURE_SUBSCRIPTION_ID and COSMOS_RESOURCE_GROUP:
        sync_cred = SyncDefaultAzureCredential()
        _ensure_capabilities(sync_cred)
        mgmt = CosmosDBManagementClient(sync_cred, AZURE_SUBSCRIPTION_ID)
        _create_database(mgmt)
        serverless = _is_serverless(mgmt)
        if serverless:
            print("  ℹ  Serverless account detected — skipping autoscale throughput settings")
        for spec in collection_specs:
            print(f"\n  Processing container '{spec['container_name']}'…")
            _create_container(
                mgmt,
                container_name=spec["container_name"],
                partition_key=spec["partition_key_path"],
                text_fields=spec["embedding_text_fields"],
                embedding_field=spec["embedding_field"],
                serverless=serverless,
            )
    else:
        print(
            "⚠  cosmos.azure_subscription_id and/or cosmos.cosmos_resource_group not set.\n"
            "   Skipping management-plane container creation.\n"
            "   Please create the containers manually before uploading documents."
        )

    # -----------------------------------------------------------------------
    # Step 2: Generate embeddings and upload documents
    # -----------------------------------------------------------------------
    print("\n🔌 Connecting to Cosmos DB…")
    async_cred = AsyncDefaultAzureCredential() if USE_RBAC_AUTH else None
    cosmos_client = (
        CosmosClient(COSMOS_ENDPOINT, credential=async_cred)
        if USE_RBAC_AUTH
        else CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    )
    embed_client = _get_embed_client()

    try:
        database = cosmos_client.get_database_client(DATABASE_NAME)

        for spec in collection_specs:
            container_name: str = spec["container_name"]
            text_fields: list[str] = spec["embedding_text_fields"]
            embedding_field: str = spec["embedding_field"]
            pk_field: str = spec["partition_key_path"].lstrip("/")
            documents: list[dict[str, Any]] = spec["documents"]

            # Verify container is accessible
            try:
                await database.get_container_client(container_name).read()
            except exceptions.CosmosResourceNotFoundError:
                print(
                    f"\n⚠  Container '{container_name}' not found in database '{DATABASE_NAME}' "
                    f"— skipping upload for this collection.\n"
                    "   Ensure management-plane creation succeeded (check subscription ID, "
                    "resource group, and Cosmos DB capabilities)."
                )
                continue

            container = database.get_container_client(container_name)
            print(f"\n📄 Generating embeddings for '{container_name}' ({len(documents)} docs)…")

            # Build embedding texts
            texts = [_build_embedding_text(doc, text_fields) for doc in documents]

            # Generate embeddings in one batch
            embeddings = await _generate_embeddings(embed_client, texts)

            # Attach embeddings and upload
            uploaded = 0
            for doc, embedding in zip(documents, embeddings):
                doc[embedding_field] = embedding
                if pk_field not in doc:
                    doc[pk_field] = doc.get("id", "")
                await container.upsert_item(doc)
                uploaded += 1

            print(f"  ✓ Uploaded {uploaded} documents to '{container_name}'")

    finally:
        await embed_client.close()
        await cosmos_client.close()
        if async_cred is not None:
            await async_cred.close()

    print("\n✅ Done — synthetic collections created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
