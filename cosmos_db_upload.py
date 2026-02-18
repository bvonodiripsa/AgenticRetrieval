"""
Cosmos DB Document Upload Script with OpenAI Embeddings

This script:
1. Automatically ensures missing Cosmos DB containers for enabled upload targets when
    management settings are available; otherwise skips missing targets.
2. Determines upload targets from config by presence of:
    `cosmos.structured_documents_root` and `cosmos.unstructured_documents_root`.
3. Supports optional `--folder` override to use one input folder for all enabled targets.
4. Scans all files in each enabled folder (content is assumed JSON), parses documents, and
    generates embeddings from the configured `llm.embed_endpoint` / `llm.embed_model`.
5. Uploads documents to the matching container and stores vectors in the `e` field.
"""

import os
import argparse
import json
import time
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from azure.cosmos import CosmosClient, exceptions
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.mgmt.cosmosdb import CosmosDBManagementClient
from azure.mgmt.cosmosdb.models import (
    SqlDatabaseCreateUpdateParameters,
    SqlDatabaseResource,
    SqlContainerCreateUpdateParameters,
    CreateUpdateOptions,
    AutoscaleSettings,
    SqlContainerResource,
    ContainerPartitionKey,
    IndexingPolicy,
    IncludedPath,
    ExcludedPath,
    VectorEmbeddingPolicy,
    VectorEmbedding,
    VectorIndex,
    VectorIndexType,
    FullTextPolicy,
    FullTextPath,
    FullTextIndexPath,
)
from openai import AzureOpenAI
from tqdm import tqdm
import requests
import yaml
from pathlib import Path

LOCAL_EMBED_WORKERS = 8
UPLOAD_WORKERS = 8
_HTTP_SESSION = requests.Session()

# Load config from yaml
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)
    
# Cosmos DB Configuration
COSMOS_ENDPOINT = CONFIG["cosmos"]["uri"]
COSMOS_KEY = CONFIG["cosmos"]["key"]
DATABASE_NAME = CONFIG["cosmos"]["database_name"]
STRUCTURED_CONTAINER_NAME = CONFIG["cosmos"]["structured_container"]
UNSTRUCTURED_CONTAINER_NAME = CONFIG["cosmos"]["unstructured_container"]
STRUCTURED_PARTITION_KEY_PATH = CONFIG["cosmos"]["structured_partition_key_path"]
UNSTRUCTURED_PARTITION_KEY_PATH = CONFIG["cosmos"]["unstructured_partition_key_path"]

# Azure Resource Manager configuration (extracted from endpoint)
# e.g., https://skf-rag-test.documents.azure.com:443/ -> account name is skf-rag-test
COSMOS_ACCOUNT_NAME = CONFIG["cosmos"]["cosmos_account_name"]
COSMOS_RESOURCE_GROUP = CONFIG["cosmos"]["cosmos_resource_group"]
AZURE_SUBSCRIPTION_ID = CONFIG["cosmos"]["azure_subscription_id"]

# Embedding configuration (single source of truth)
EMBED_ENDPOINT = str(CONFIG["llm"]["embed_endpoint"]).strip().strip('"')
EMBED_MODEL = CONFIG["llm"]["embed_model"]
EMBEDDING_DIMENSIONS = int(CONFIG["llm"]["embed_dimensions"])
LLM_API_VERSION = CONFIG["llm"]["api_version"]
AZURE_OPENAI_KEY = CONFIG["llm"]["azure_openai_key"]

# Batch configuration
EMBEDDING_BATCH_SIZE = int(CONFIG["cosmos"]["embedding_batch_size"])  # Number of texts to embed in one API call



# Path to JSON documents
UNSTRUCTURED_DOCUMENTS_ROOT = CONFIG["cosmos"]["unstructured_documents_root"]
STRUCTURED_DOCUMENTS_ROOT = CONFIG["cosmos"]["structured_documents_root"]

# ============== INDEXING POLICY (CONFIG-DRIVEN) ==============
INDEXING_POLICIES = {
    "structured": json.loads(CONFIG["cosmos"]["indexing_policies_json"]["structured"]),
    "unstructured": json.loads(CONFIG["cosmos"]["indexing_policies_json"]["unstructured"]),
}
FULL_TEXT_POLICIES = {
    "structured": json.loads(CONFIG["cosmos"]["full_text_policies_json"]["structured"]),
    "unstructured": json.loads(CONFIG["cosmos"]["full_text_policies_json"]["unstructured"]),
}
VECTOR_EMBEDDING_POLICY = json.loads(CONFIG["cosmos"]["vector_embedding_policy_json"])


def _normalize_embedding(embedding: List[float]) -> List[float]:
    values = [float(x) for x in embedding]
    if len(values) > EMBEDDING_DIMENSIONS:
        return values[:EMBEDDING_DIMENSIONS]
    if len(values) < EMBEDDING_DIMENSIONS:
        return values + [0.0] * (EMBEDDING_DIMENSIONS - len(values))
    return values


def get_embedding_client() -> AzureOpenAI | None:
    """Initialize a single embedding client from llm.embed_endpoint/embed_model settings."""
    endpoint = EMBED_ENDPOINT.rstrip("/")
    if endpoint.endswith("/api/embeddings"):
        return None

    azure_endpoint = endpoint
    if "/openai/deployments/" in azure_endpoint:
        azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]

    if not AZURE_OPENAI_KEY or not str(AZURE_OPENAI_KEY).strip():
        raise ValueError("llm.azure_openai_key must be set for Azure OpenAI embedding endpoint")

    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=AZURE_OPENAI_KEY,
        api_version=LLM_API_VERSION,
    )


def get_cosmos_client(use_rbac_auth: bool, credential=None) -> CosmosClient:
    """Initialize and return Cosmos DB client for selected auth mode."""
    if not COSMOS_ENDPOINT:
        raise ValueError("Cosmos DB endpoint not configured. Set COSMOS_ENDPOINT.")
    
    if use_rbac_auth:
        # Use Entra ID RBAC authentication (DefaultAzureCredential)
        if credential is None:
            credential = DefaultAzureCredential()
        print("✓ Using Entra ID RBAC authentication for Cosmos DB")
        return CosmosClient(COSMOS_ENDPOINT, credential=credential)
    else:
        # Use key-based authentication
        if not COSMOS_KEY:
            raise ValueError("Cosmos DB key not configured. Set cosmos.key in config.yaml.")
        print("✓ Using key-based authentication for Cosmos DB")
        return CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)


def extract_account_name_from_endpoint(endpoint: str) -> str:
    """Extract the Cosmos DB account name from the endpoint URL."""
    # https://skf-rag-test.documents.azure.com:443/ -> skf-rag-test
    match = re.match(r'https://([^.]+)\.documents\.azure\.com', endpoint)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract account name from endpoint: {endpoint}")


def enable_vector_search_capability(credential, subscription_id: str, resource_group: str, account_name: str):
    """
    Enable Vector Search capability on the Cosmos DB account if not already enabled.
    This is required before creating containers with vector indexing.
    """
    from azure.mgmt.cosmosdb.models import DatabaseAccountUpdateParameters, Capability
    
    mgmt_client = CosmosDBManagementClient(credential, subscription_id)
    
    # Get current account
    account = mgmt_client.database_accounts.get(resource_group, account_name)
    
    # Check if VectorSearch is already enabled
    current_capabilities = account.capabilities or []
    capability_names = [c.name for c in current_capabilities]
    
    capabilities_to_add = []
    
    if "EnableNoSQLVectorSearch" not in capability_names:
        print("  Enabling NoSQL Vector Search capability...")
        capabilities_to_add.append(Capability(name="EnableNoSQLVectorSearch"))
    
    if "EnableNoSQLFullTextSearch" not in capability_names:
        print("  Enabling NoSQL Full Text Search capability...")
        capabilities_to_add.append(Capability(name="EnableNoSQLFullTextSearch"))
    
    if capabilities_to_add:
        # Add new capabilities to existing ones
        all_capabilities = list(current_capabilities) + capabilities_to_add
        
        update_params = DatabaseAccountUpdateParameters(
            capabilities=all_capabilities
        )
        
        poller = mgmt_client.database_accounts.begin_update(
            resource_group_name=resource_group,
            account_name=account_name,
            update_parameters=update_params
        )
        print("  Waiting for capabilities to be enabled (this may take a few minutes)...")
        poller.result()
        print("  ✓ Vector Search and Full Text Search capabilities enabled")
    else:
        print("  ✓ Vector Search and Full Text Search capabilities already enabled")


def create_database_and_container_via_management(credential, enabled_policy_keys: List[str]):
    """
    Create database and configured containers using Azure Resource Manager (control plane).
    Existing containers are never updated here; only missing containers are created.
    This uses Azure RBAC permissions instead of Cosmos DB data plane RBAC.
    """
    # Get configuration
    subscription_id = AZURE_SUBSCRIPTION_ID
    resource_group = COSMOS_RESOURCE_GROUP
    account_name = COSMOS_ACCOUNT_NAME or extract_account_name_from_endpoint(COSMOS_ENDPOINT)
    
    if not subscription_id or not resource_group:
        raise ValueError(
            "Azure subscription ID and resource group are required for management operations. "
            "Set cosmos.azure_subscription_id and cosmos.cosmos_resource_group in config.yaml."
        )
    
    print(f"  Account: {account_name}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Subscription: {subscription_id}")
    
    # Enable Vector Search and Full Text Search capabilities if needed
    print("\n  Checking account capabilities...")
    enable_vector_search_capability(credential, subscription_id, resource_group, account_name)
    
    # Create management client
    mgmt_client = CosmosDBManagementClient(credential, subscription_id)
    
    # Create database
    print(f"\n  Creating database '{DATABASE_NAME}'...")
    try:
        db_params = SqlDatabaseCreateUpdateParameters(
            resource=SqlDatabaseResource(id=DATABASE_NAME)
        )
        poller = mgmt_client.sql_resources.begin_create_update_sql_database(
            resource_group_name=resource_group,
            account_name=account_name,
            database_name=DATABASE_NAME,
            create_update_sql_database_parameters=db_params
        )
        poller.result()  # Wait for completion
        print(f"  ✓ Database '{DATABASE_NAME}' ready")
    except Exception as e:
        if "Conflict" in str(e) or "already exists" in str(e).lower():
            print(f"  ✓ Database '{DATABASE_NAME}' already exists")
        else:
            print(f"  Error creating database: {e}")
            raise
    
    def container_exists(container_name: str) -> bool:
        try:
            mgmt_client.sql_resources.get_sql_container(
                resource_group_name=resource_group,
                account_name=account_name,
                database_name=DATABASE_NAME,
                container_name=container_name,
            )
            return True
        except HttpResponseError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            raise

    container_name_by_policy_key = {
        "structured": STRUCTURED_CONTAINER_NAME,
        "unstructured": UNSTRUCTURED_CONTAINER_NAME,
    }
    partition_key_path_by_policy_key = {
        "structured": STRUCTURED_PARTITION_KEY_PATH,
        "unstructured": UNSTRUCTURED_PARTITION_KEY_PATH,
    }
    container_specs = [
        (policy_key, container_name_by_policy_key[policy_key])
        for policy_key in enabled_policy_keys
        if policy_key in container_name_by_policy_key
    ]
    processed_names = set()
    max_retries = 5
    retry_delay = 30  # seconds

    for policy_key, container_name in container_specs:
        if not container_name or not str(container_name).strip():
            print(f"  ⚠ Skipping {policy_key}: container name is not configured")
            continue
        if container_name in processed_names:
            continue
        processed_names.add(container_name)

        if container_exists(container_name):
            print(f"  ✓ Container '{container_name}' already exists - skipping")
            continue

        partition_key_path = partition_key_path_by_policy_key[policy_key]
        if not partition_key_path or not str(partition_key_path).strip():
            print(f"  ⚠ Skipping {policy_key}: partition key path is not configured")
            continue

        fts_policy_cfg = FULL_TEXT_POLICIES[policy_key]
        index_policy_cfg = INDEXING_POLICIES[policy_key]

        indexing_policy = IndexingPolicy(
            indexing_mode=index_policy_cfg["indexingMode"],
            automatic=index_policy_cfg["automatic"],
            included_paths=[IncludedPath(path=p["path"]) for p in index_policy_cfg["includedPaths"]],
            excluded_paths=[ExcludedPath(path=p["path"]) for p in index_policy_cfg["excludedPaths"]],
            vector_indexes=[
                VectorIndex(path=v["path"], type=VectorIndexType(v["type"]))
                for v in index_policy_cfg["vectorIndexes"]
            ],
            full_text_indexes=[
                FullTextIndexPath(path=f["path"])
                for f in index_policy_cfg["fullTextIndexes"]
            ]
        )

        vector_embedding_policy = VectorEmbeddingPolicy(
            vector_embeddings=[
                VectorEmbedding(
                    path=v["path"],
                    data_type=v["dataType"],
                    dimensions=v["dimensions"],
                    distance_function=v["distanceFunction"]
                )
                for v in VECTOR_EMBEDDING_POLICY["vectorEmbeddings"]
            ]
        )

        full_text_policy = FullTextPolicy(
            default_language=fts_policy_cfg["defaultLanguage"],
            full_text_paths=[
                FullTextPath(path=p["path"], language=p["language"])
                for p in fts_policy_cfg["fullTextPaths"]
            ]
        )

        print(f"\n  Creating container '{container_name}'...")
        container_resource = SqlContainerResource(
            id=container_name,
            partition_key=ContainerPartitionKey(
                paths=[partition_key_path],
                kind="Hash"
            ),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy
        )
        container_params = SqlContainerCreateUpdateParameters(resource=container_resource)
        tried_dedicated_autoscale = False

        for attempt in range(max_retries):
            try:
                # Safety guard: never run create/update against an existing container.
                if container_exists(container_name):
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break

                poller = mgmt_client.sql_resources.begin_create_update_sql_container(
                    resource_group_name=resource_group,
                    account_name=account_name,
                    database_name=DATABASE_NAME,
                    container_name=container_name,
                    create_update_sql_container_parameters=container_params
                )
                poller.result()
                print(f"  ✓ Container '{container_name}' created")
                break
            except Exception as e:
                error_str = str(e)
                if (
                    "Vector Indexing is not supported for shared throughput offer" in error_str
                    and not tried_dedicated_autoscale
                ):
                    print(
                        "  ⚠ Shared throughput doesn't support vector indexing for this container; "
                        "retrying with dedicated autoscale throughput (1000 RU min)..."
                    )
                    container_params = SqlContainerCreateUpdateParameters(
                        resource=container_resource,
                        options=CreateUpdateOptions(
                            autoscale_settings=AutoscaleSettings(max_throughput=1000)
                        )
                    )
                    tried_dedicated_autoscale = True
                    continue
                if "Conflict" in error_str or "already exists" in error_str.lower():
                    print(f"  ✓ Container '{container_name}' already exists - using as-is (no settings update)")
                    break
                if "capability has not been enabled" in error_str.lower() and attempt < max_retries - 1:
                    print(f"  ⏳ Waiting for capabilities to propagate (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                print(f"  Error creating container '{container_name}': {e}")
                raise

def generate_embedding_text_structured(doc: Dict[str, Any]) -> str:
    """
    Generate the text to be embedded from a document.
    Combines all relevant text fields into a single string.
    """
    text_parts = []
    
    # Add title if present
    if "title" in doc and doc["title"]:
        text_parts.append(f"Title: {doc['title']}")
    
    # Add summary if present
    if "summary" in doc and doc["summary"]:
        text_parts.append(f"Summary: {doc['summary']}")
    
    # Add content if present
    if "content" in doc and doc["content"]:
        text_parts.append(f"Content: {doc['content']}")
        
    # Add designation if present
    if "designation" in doc and doc["designation"]:
        text_parts.append(f"Designation: {doc['designation']}")
    
    # Add description if present
    if "description" in doc and doc["description"]:
        text_parts.append(f"Description: {doc['description']}")
    
    # Add long_description if present
    if "long_description" in doc and doc["long_description"]:
        text_parts.append(f"Long Description: {doc['long_description']}")
    
    # Add benefits if present
    if "benefits" in doc and doc["benefits"]:
        benefits = doc["benefits"]
        if isinstance(benefits, list):
            text_parts.append(f"Benefits: {', '.join(benefits)}")
        else:
            text_parts.append(f"Benefits: {benefits}")
    
    # Add tags if present
    if "tags" in doc and doc["tags"]:
        tags = doc["tags"]
        if isinstance(tags, list):
            text_parts.append(f"Tags: {', '.join(tags)}")
        else:
            text_parts.append(f"Tags: {tags}")
    
    # Add taxonomy if present
    if "taxonomy" in doc and doc["taxonomy"]:
        text_parts.append(f"Taxonomy: {doc['taxonomy']}")

    
    # If no specific fields found, use the entire JSON
    if not text_parts:
        text_parts.append(json.dumps(doc, ensure_ascii=False))
    
    return "\n".join(text_parts)

def generate_embedding_text_unstructured(doc: Dict[str, Any]) -> str:
    """
    Generate the text to be embedded from a document.
    Combines all relevant text fields into a single string.
    """
    text_parts = []
    
    # Add title, summary, and content if present
    if "title" in doc and doc["title"]:
        text_parts.append(f"Title: {doc['title']}")
    if "summary" in doc and doc["summary"]:
        text_parts.append(f"Summary: {doc['summary']}")
    if "content" in doc and doc["content"]:
        text_parts.append(f"Content: {doc['content']}")
    return "\n".join(text_parts)


def generate_embeddings_batch(embed_client: AzureOpenAI | None, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts from a single configured endpoint/model."""
    if EMBED_ENDPOINT.rstrip("/").endswith("/api/embeddings"):
        batch_endpoint = EMBED_ENDPOINT.rstrip("/")
        if batch_endpoint.endswith("/api/embeddings"):
            batch_endpoint = batch_endpoint[: -len("/api/embeddings")] + "/api/embed"

        # Fast path for Ollama /api/embed with batched input
        try:
            response = _HTTP_SESSION.post(
                batch_endpoint,
                json={"model": EMBED_MODEL, "input": texts},
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings_payload = payload.get("embeddings")
            if isinstance(embeddings_payload, list) and len(embeddings_payload) == len(texts):
                return [_normalize_embedding(embedding) for embedding in embeddings_payload]
        except Exception:
            # Fallback to parallel single-item /api/embeddings requests
            pass

        def embed_single(text: str) -> List[float]:
            response = _HTTP_SESSION.post(
                EMBED_ENDPOINT,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            if "embedding" not in payload:
                raise ValueError(f"Unexpected embedding response payload: {payload}")
            return _normalize_embedding(payload["embedding"])

        workers = min(max(1, LOCAL_EMBED_WORKERS), len(texts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(embed_single, texts))

    if embed_client is None:
        raise ValueError("Embedding client is not initialized for Azure OpenAI endpoint")

    response = embed_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [_normalize_embedding(item.embedding) for item in response.data]


def find_all_input_files(root_path: str) -> List[str]:
    """Find all files in the directory tree (content is expected to be JSON)."""
    input_files = []
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and the script itself
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.startswith('.'):
                input_files.append(os.path.join(root, file))
    
    return input_files


def load_json_document(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON document from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def replace_document_id(doc: Dict[str, Any], relative_path: str) -> Dict[str, Any]:
    """Replace any existing id with a deterministic generated id based on relative path only."""
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    generated_id = f"doc-{digest}"
    doc["id"] = generated_id
    return doc


def upload_document(container, doc: Dict[str, Any]) -> bool:
    """Upload a single document to Cosmos DB."""
    try:
        container.upsert_item(doc)
        return True
    except exceptions.CosmosHttpResponseError as e:
        doc_id = doc["id"] if "id" in doc else "unknown"
        print(f"Error uploading document {doc_id}: {e}")
        return False


def upload_documents_batch(container, docs: List[Dict[str, Any]]) -> tuple[int, int]:
    """Upload a batch of docs in parallel. Returns (success_count, failed_count)."""
    if not docs:
        return 0, 0

    workers = min(max(1, UPLOAD_WORKERS), len(docs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda item: upload_document(container, item), docs))
    success_count = sum(1 for result in results if result)
    return success_count, len(results) - success_count


def main():
    """Main function to orchestrate the upload process."""
    parser = argparse.ArgumentParser(description="Upload JSON documents with embeddings to Cosmos DB")
    parser.add_argument(
        "--folder",
        "--documents-root",
        dest="documents_root",
        help="Path to folder containing JSON files to process (overrides config.yaml path selection)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Cosmos DB Document Upload with OpenAI Embeddings")
    print("=" * 60)
    
    # Validate configuration
    print("\n📋 Checking configuration...")
    
    if not COSMOS_ENDPOINT:
        print("❌ Error: Cosmos DB endpoint not configured.")
        print("   Please set cosmos.uri in config.yaml.")
        print("   Example: cosmos.uri: https://your-account.documents.azure.com:443/")
        return
    
    if not COSMOS_KEY:
        print("❌ Error: Cosmos DB key not configured.")
        print("   Please set cosmos.key in config.yaml.")
        return
    
    if not EMBED_ENDPOINT:
        print("❌ Error: Embedding endpoint not configured.")
        print("   Please set llm.embed_endpoint in config.yaml.")
        return

    if not EMBED_MODEL:
        print("❌ Error: Embedding model not configured.")
        print("   Please set llm.embed_model in config.yaml.")
        return

    if not EMBED_ENDPOINT.rstrip("/").endswith("/api/embeddings") and (not AZURE_OPENAI_KEY or not str(AZURE_OPENAI_KEY).strip()):
        print("❌ Error: Azure OpenAI API key not configured.")
        print("   Please set llm.azure_openai_key in config.yaml.")
        return

    print(f"✓ Embedding endpoint: {EMBED_ENDPOINT}")
    print(f"✓ Embedding model: {EMBED_MODEL}")
    print(f"✓ Embedding dimensions: {EMBEDDING_DIMENSIONS}")

    def has_value(value: Any) -> bool:
        return bool(value and str(value).strip())

    upload_targets = []
    if has_value(STRUCTURED_DOCUMENTS_ROOT):
        upload_targets.append({
            "name": "structured",
            "documents_root": STRUCTURED_DOCUMENTS_ROOT,
            "container_name": STRUCTURED_CONTAINER_NAME,
            "partition_key_path": STRUCTURED_PARTITION_KEY_PATH,
            "text_builder": generate_embedding_text_structured,
        })
    else:
        print("⚠ Skipping structured upload: cosmos.structured_documents_root is empty or not set.")

    if has_value(UNSTRUCTURED_DOCUMENTS_ROOT):
        upload_targets.append({
            "name": "unstructured",
            "documents_root": UNSTRUCTURED_DOCUMENTS_ROOT,
            "container_name": UNSTRUCTURED_CONTAINER_NAME,
            "partition_key_path": UNSTRUCTURED_PARTITION_KEY_PATH,
            "text_builder": generate_embedding_text_unstructured,
        })
    else:
        print("⚠ Skipping unstructured upload: cosmos.unstructured_documents_root is empty or not set.")

    if not upload_targets:
        print("No upload targets configured. Nothing to upload.")
        return

    if args.documents_root:
        if not str(args.documents_root).strip():
            print("❌ Error: --folder must have a non-empty value.")
            return
        for target in upload_targets:
            target["documents_root"] = args.documents_root

    print("\n🔌 Initializing clients...")
    cosmos_client = get_cosmos_client(use_rbac_auth=False)
    print("✓ Cosmos client initialized (key auth)")

    # Initialize embedding client (single endpoint/model)
    embed_client = get_embedding_client()
    print("✓ Embedding client initialized")

    database = cosmos_client.get_database_client(DATABASE_NAME)
    print(f"✓ Connected to database '{DATABASE_NAME}'")

    def container_exists_data_plane(container_name: str) -> bool:
        try:
            database.get_container_client(container_name).read()
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    missing_policy_keys = [
        target["name"]
        for target in upload_targets
        if not container_exists_data_plane(target["container_name"])
    ]

    if missing_policy_keys:
        if AZURE_SUBSCRIPTION_ID and COSMOS_RESOURCE_GROUP:
            try:
                credential = DefaultAzureCredential()
                print("\n📦 Missing containers detected; attempting management-plane create...")
                create_database_and_container_via_management(credential, missing_policy_keys)
            except Exception as e:
                print(f"⚠ Container auto-create failed: {e}")
        else:
            print("\n⚠ Missing containers detected, but management settings are incomplete.")
            print("   Set cosmos.azure_subscription_id and cosmos.cosmos_resource_group to auto-create missing containers.")

    upload_targets_ready = []
    for target in upload_targets:
        if container_exists_data_plane(target["container_name"]):
            upload_targets_ready.append(target)
        else:
            print(f"⚠ Skipping {target['name']} upload: container '{target['container_name']}' does not exist.")

    if not upload_targets_ready:
        print("No existing target containers available for upload. Nothing to do.")
        return

    batch_size = EMBEDDING_BATCH_SIZE
    successful_uploads = 0
    failed_uploads = 0
    total_files_seen = 0
    uploaded_containers = []
    total_parse_seconds = 0.0
    total_embed_seconds = 0.0
    total_upload_seconds = 0.0

    for target in upload_targets_ready:
        target_name = target["name"]
        documents_root = target["documents_root"]
        container_name = target["container_name"]
        partition_key_path = str(target["partition_key_path"])
        build_embedding_text = target["text_builder"]

        if not os.path.isdir(documents_root):
            print(f"⚠ Skipping {target_name} upload: folder does not exist: {documents_root}")
            continue

        container = database.get_container_client(container_name)
        uploaded_containers.append(container_name)
        print(f"✓ Connected to container '{container_name}' for {target_name} upload")
        print(f"\n🔍 Scanning for files in ({target_name}): {documents_root}")
        input_files = find_all_input_files(documents_root)
        total_files_seen += len(input_files)
        print(f"✓ Found {len(input_files)} files")

        if not input_files:
            print(f"⚠ No files found for {target_name}. Skipping.")
            continue

        print(f"\n📄 Processing {target_name} documents (batch size: {batch_size})...")
        batch_docs = []
        batch_texts = []

        for file_path in tqdm(input_files, desc=f"Processing {target_name} files"):
            parse_start = time.perf_counter()
            doc = load_json_document(file_path)
            if doc is None:
                failed_uploads += 1
                total_parse_seconds += (time.perf_counter() - parse_start)
                continue

            relative_path = os.path.relpath(file_path, documents_root)
            doc['_source_file'] = relative_path

            doc = replace_document_id(doc, relative_path)

            embedding_text = build_embedding_text(doc)
            batch_docs.append(doc)
            batch_texts.append(embedding_text)
            total_parse_seconds += (time.perf_counter() - parse_start)

            if len(batch_docs) >= batch_size:
                try:
                    embed_start = time.perf_counter()
                    embeddings = generate_embeddings_batch(embed_client, batch_texts)
                    total_embed_seconds += (time.perf_counter() - embed_start)
                    for item_doc, embedding in zip(batch_docs, embeddings):
                        item_doc['e'] = embedding

                    upload_start = time.perf_counter()
                    success_count, failed_count = upload_documents_batch(container, batch_docs)
                    total_upload_seconds += (time.perf_counter() - upload_start)
                    successful_uploads += success_count
                    failed_uploads += failed_count
                except Exception as e:
                    print(f"\nError processing {target_name} batch: {e}")
                    failed_uploads += len(batch_docs)

                batch_docs = []
                batch_texts = []

        if batch_docs:
            try:
                embed_start = time.perf_counter()
                embeddings = generate_embeddings_batch(embed_client, batch_texts)
                total_embed_seconds += (time.perf_counter() - embed_start)
                for item_doc, embedding in zip(batch_docs, embeddings):
                    item_doc['e'] = embedding

                upload_start = time.perf_counter()
                success_count, failed_count = upload_documents_batch(container, batch_docs)
                total_upload_seconds += (time.perf_counter() - upload_start)
                successful_uploads += success_count
                failed_uploads += failed_count
            except Exception as e:
                print(f"\nError processing final {target_name} batch: {e}")
                failed_uploads += len(batch_docs)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Upload Summary")
    print("=" * 60)
    print(f"✓ Successfully uploaded: {successful_uploads}")
    print(f"✗ Failed: {failed_uploads}")
    print(f"📁 Total files scanned: {total_files_seen}")
    print(f"🗄️  Database: {DATABASE_NAME}")
    print(f"📦 Containers touched: {', '.join(uploaded_containers) if uploaded_containers else 'none'}")
    print(f"⏱ Parse/build text time: {total_parse_seconds:.1f}s")
    print(f"⏱ Embedding time: {total_embed_seconds:.1f}s")
    print(f"⏱ Upload time: {total_upload_seconds:.1f}s")
    if successful_uploads > 0:
        print(f"⏱ Avg embed+upload per uploaded doc: {(total_embed_seconds + total_upload_seconds) / successful_uploads:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
