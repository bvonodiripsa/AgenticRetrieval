"""
Create Synthetic Cosmos DB Collections

This script creates a Cosmos DB database and two small synthetic collections
populated with autoscale throughput for use in tests.

Collection 1: "articles"
  Properties: title, author, category, body  (all full-text indexed)

Collection 2: "products"
  Properties: name, brand, description, features  (all full-text indexed)

Both collections have a vector embedding property "e" (float32, 1024-dim,
cosine, diskANN) whose value is generated from the concatenation of all four
text fields, using the same Azure OpenAI embedding model and configuration
defined in config.yaml (embedding section).

Usage:
    python tests/create_synthetic_collections.py

The script reads credentials and endpoint settings from config.yaml in the
repository root, following the same pattern as cosmos_db_upload.py.
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
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

_COSMOS_CFG = CONFIG.get("cosmos", {})
# Embedding section values take precedence over the llm section (mirrors cosmos_db_upload.py).
_EMBED_CFG = {**CONFIG.get("llm", {}), **CONFIG.get("embedding", {})}

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
# The embedding endpoint shares the LLM api_version (same Azure OpenAI service).
EMBED_API_VERSION: str = str(CONFIG.get("llm", {}).get("api_version", "2024-05-01-preview"))
EMBED_API_KEY: str = str(
    _EMBED_CFG.get("embed_api_key") or CONFIG.get("llm", {}).get("azure_openai_key", "") or ""
).strip()

# Autoscale max throughput (RU/s) for each container
AUTOSCALE_MAX_THROUGHPUT = 1000

# ---------------------------------------------------------------------------
# Collection definitions
# ---------------------------------------------------------------------------

# Each collection spec:
#   name            – Cosmos DB container name
#   partition_key   – partition key path
#   text_fields     – the four text properties (also full-text indexed)
#   documents       – small synthetic dataset

COLLECTION_SPECS: list[dict[str, Any]] = [
    {
        "name": "articles",
        "partition_key": "/pk",
        "text_fields": ["title", "author", "category", "body"],
        "documents": [
            {
                "pk": "article-1",
                "title": "Advances in Quantum Computing",
                "author": "Alice Nguyen",
                "category": "Technology",
                "body": (
                    "Quantum computing leverages quantum mechanical phenomena such as "
                    "superposition and entanglement to perform computations. Recent "
                    "breakthroughs have demonstrated error-corrected logical qubits."
                ),
            },
            {
                "pk": "article-2",
                "title": "Climate Change and Renewable Energy",
                "author": "Bob Martinez",
                "category": "Environment",
                "body": (
                    "Transitioning to renewable energy sources is critical for mitigating "
                    "climate change. Solar and wind power capacity has grown exponentially "
                    "over the past decade, driven by falling costs and policy incentives."
                ),
            },
            {
                "pk": "article-3",
                "title": "The Future of Artificial Intelligence",
                "author": "Carol Smith",
                "category": "Technology",
                "body": (
                    "Large language models and multimodal AI systems are reshaping how "
                    "software is built. Researchers are exploring alignment, safety, and "
                    "interpretability as AI capabilities continue to expand rapidly."
                ),
            },
            {
                "pk": "article-4",
                "title": "Deep-Sea Exploration Discoveries",
                "author": "David Lee",
                "category": "Science",
                "body": (
                    "Recent expeditions to the Mariana Trench have uncovered previously "
                    "unknown species. Advanced underwater robotics enable scientists to "
                    "study ecosystems under extreme pressure and darkness."
                ),
            },
            {
                "pk": "article-5",
                "title": "Urban Farming and Food Security",
                "author": "Eva Chen",
                "category": "Agriculture",
                "body": (
                    "Vertical farming and hydroponics allow crops to be grown in urban "
                    "environments with minimal water use. These methods could help ensure "
                    "food security as global population continues to grow."
                ),
            },
            {
                "pk": "article-6",
                "title": "Breakthroughs in Gene Therapy",
                "author": "Frank Osei",
                "category": "Medicine",
                "body": (
                    "CRISPR-based gene therapies have entered clinical trials for several "
                    "genetic disorders. Early results show promise for conditions like "
                    "sickle-cell disease and certain inherited blindness syndromes."
                ),
            },
            {
                "pk": "article-7",
                "title": "Space Tourism: The Next Frontier",
                "author": "Grace Kim",
                "category": "Space",
                "body": (
                    "Commercial space companies are making suborbital and orbital tourism "
                    "a reality. Costs are expected to decline as reusable rockets mature "
                    "and competition in the launch market intensifies."
                ),
            },
            {
                "pk": "article-8",
                "title": "Blockchain Beyond Cryptocurrency",
                "author": "Hiro Tanaka",
                "category": "Technology",
                "body": (
                    "Distributed ledger technology is being applied to supply chain "
                    "management, healthcare records, and digital identity systems. Smart "
                    "contracts automate complex multi-party agreements without intermediaries."
                ),
            },
            {
                "pk": "article-9",
                "title": "The Psychology of Decision Making",
                "author": "Iris Patel",
                "category": "Psychology",
                "body": (
                    "Cognitive biases influence how people make choices under uncertainty. "
                    "Behavioral economics integrates psychological insights with economic "
                    "models to better predict and guide human decision behavior."
                ),
            },
            {
                "pk": "article-10",
                "title": "Ancient Roman Engineering Marvels",
                "author": "James Rivera",
                "category": "History",
                "body": (
                    "Roman aqueducts transported water hundreds of kilometers using only "
                    "gravity. The engineering principles behind arches and concrete "
                    "construction enabled structures that endure two millennia later."
                ),
            },
        ],
    },
    {
        "name": "products",
        "partition_key": "/pk",
        "text_fields": ["name", "brand", "description", "features"],
        "documents": [
            {
                "pk": "product-1",
                "name": "UltraComfort Ergonomic Chair",
                "brand": "ErgoTech",
                "description": "Premium office chair designed for all-day comfort with lumbar support.",
                "features": "Adjustable armrests, breathable mesh back, tilt tension control, 5-year warranty",
            },
            {
                "pk": "product-2",
                "name": "ProMax Wireless Headphones",
                "brand": "SoundWave",
                "description": "Over-ear noise-cancelling headphones with 40-hour battery life.",
                "features": "Active noise cancellation, Bluetooth 5.3, foldable design, USB-C charging",
            },
            {
                "pk": "product-3",
                "name": "SmartHome Hub 3000",
                "brand": "NexusTech",
                "description": "Central controller for all your smart home devices with voice assistant support.",
                "features": "Works with Alexa and Google Home, Zigbee/Z-Wave/Wi-Fi, touchscreen display",
            },
            {
                "pk": "product-4",
                "name": "TrailBlazer Hiking Boots",
                "brand": "Summit Gear",
                "description": "Waterproof leather hiking boots with superior ankle support.",
                "features": "Gore-Tex lining, Vibram outsole, full-grain leather, wide toe box option",
            },
            {
                "pk": "product-5",
                "name": "AquaPure Water Filter Pitcher",
                "brand": "ClearFlow",
                "description": "7-stage filtration pitcher that removes 99% of contaminants.",
                "features": "NSF certified, BPA-free, 10-cup capacity, filter life indicator",
            },
            {
                "pk": "product-6",
                "name": "SwiftCharge 65W GaN Charger",
                "brand": "VoltEdge",
                "description": "Compact gallium nitride charger with three ports for fast charging.",
                "features": "3 USB-C + 1 USB-A ports, foldable prongs, travel-ready size, PD 3.0",
            },
            {
                "pk": "product-7",
                "name": "FocusLens 4K Webcam",
                "brand": "PixelView",
                "description": "Ultra-HD webcam with AI auto-framing for professional video calls.",
                "features": "4K 30fps or 1080p 60fps, low-light correction, dual stereo microphones",
            },
            {
                "pk": "product-8",
                "name": "NightSky Pro Telescope",
                "brand": "AstroVision",
                "description": "Computerized telescope with 90,000-object database for stargazers.",
                "features": "5-inch aperture, GoTo mount, smartphone adapter, carry case included",
            },
            {
                "pk": "product-9",
                "name": "ZenBreath Air Purifier",
                "brand": "CleanAir Co.",
                "description": "HEPA air purifier covering up to 500 sq ft with real-time air quality display.",
                "features": "True HEPA H13 filter, activated carbon layer, whisper-quiet mode, auto mode",
            },
            {
                "pk": "product-10",
                "name": "PowerPedal Stationary Bike",
                "brand": "FitCycle",
                "description": "Magnetic resistance indoor cycling bike with 32 resistance levels.",
                "features": "Quiet belt drive, adjustable seat and handlebars, built-in Bluetooth, LCD display",
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Indexing / policy helpers
# ---------------------------------------------------------------------------


def _build_indexing_policy(text_fields: list[str]) -> dict[str, Any]:
    """Return the indexing policy dict for a collection with the given text fields."""
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/\"_etag\"/?"},
            {"path": "/e/*"},
        ],
        "fullTextIndexes": [{"path": f"/{f}"} for f in text_fields],
        "vectorIndexes": [
            {
                "path": "/e",
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


_VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/e",
            "dataType": "float32",
            "dimensions": EMBED_DIMENSIONS,
            "distanceFunction": "cosine",
        }
    ]
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


def _create_container(
    mgmt: CosmosDBManagementClient,
    container_name: str,
    partition_key: str,
    text_fields: list[str],
) -> None:
    account_name = _get_account_name()
    if _container_exists_mgmt(mgmt, container_name):
        print(f"  ✓ Container '{container_name}' already exists — skipping creation")
        return

    ip_cfg = _build_indexing_policy(text_fields)
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
                path=v["path"],
                data_type=v["dataType"],
                dimensions=v["dimensions"],
                distance_function=v["distanceFunction"],
            )
            for v in _VECTOR_EMBEDDING_POLICY["vectorEmbeddings"]
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
            "embed_api_key (or azure_openai_key) must be set in config.yaml for embedding generation"
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
        print("❌ cosmos.uri is not set in config.yaml")
        sys.exit(1)
    if not USE_RBAC_AUTH and not COSMOS_KEY:
        print("❌ cosmos.key is not set (or set cosmos.use_rbac_auth: true)")
        sys.exit(1)
    if not EMBED_ENDPOINT:
        print("❌ embedding.embed_endpoint is not set in config.yaml")
        sys.exit(1)
    if not EMBED_MODEL:
        print("❌ embedding.embed_model is not set in config.yaml")
        sys.exit(1)
    if not EMBED_API_KEY:
        print("❌ embedding.embed_api_key is not set in config.yaml")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Create database and containers via management plane
    # -----------------------------------------------------------------------
    print("\n📦 Setting up database and containers…")

    if AZURE_SUBSCRIPTION_ID and COSMOS_RESOURCE_GROUP:
        sync_cred = SyncDefaultAzureCredential()
        _ensure_capabilities(sync_cred)
        mgmt = CosmosDBManagementClient(sync_cred, AZURE_SUBSCRIPTION_ID)
        _create_database(mgmt)
        for spec in COLLECTION_SPECS:
            print(f"\n  Processing container '{spec['name']}'…")
            _create_container(
                mgmt,
                container_name=spec["name"],
                partition_key=spec["partition_key"],
                text_fields=spec["text_fields"],
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

        for spec in COLLECTION_SPECS:
            container_name: str = spec["name"]
            text_fields: list[str] = spec["text_fields"]
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
                doc_with_embedding = {**doc, "e": embedding}
                # Use the pk value as the document id for determinism
                doc_with_embedding["id"] = doc["pk"]
                await container.upsert_item(doc_with_embedding)
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
