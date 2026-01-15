"""
Amazon S3 Vectors Example

This example demonstrates:
1. Creating a vector bucket
2. Creating a vector index with specific dimensions and distance metric
3. Generating embeddings with Amazon Bedrock Titan
4. Inserting vectors with metadata
5. Querying vectors with similarity search
6. Filtered queries using metadata

Requirements:
    pip install boto3

AWS Setup:
    - Configure AWS credentials (aws configure)
    - Ensure you have access to S3 Vectors (preview)
    - Enable Amazon Bedrock and Titan model access
"""

import boto3
import json
import time
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

AWS_REGION = "us-east-1"  # S3 Vectors availability varies by region

VECTOR_BUCKET_NAME = "my-vector-embeddings"
INDEX_NAME = "documents"

# Amazon Titan Text Embeddings V2 outputs 1024 dimensions by default
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024


# =============================================================================
# Client Setup
# =============================================================================

def get_clients():
    """Initialize S3 Vectors and Bedrock clients."""
    s3vectors = boto3.client("s3vectors", region_name=AWS_REGION)
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return s3vectors, bedrock


# =============================================================================
# Embedding Generation
# =============================================================================

class BedrockEmbedder:
    """Generate embeddings using Amazon Bedrock Titan."""

    def __init__(self, bedrock_client, model_id: str = EMBEDDING_MODEL_ID):
        self.bedrock = bedrock_client
        self.model_id = model_id

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_single(text)
            embeddings.append(embedding)
        return embeddings


class LocalEmbedder:
    """
    Alternative: Use sentence-transformers locally.
    Useful for testing without Bedrock costs.
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Use a 1024-dim model to match Titan
            self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def embed_single(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


# =============================================================================
# Vector Bucket Operations
# =============================================================================

def create_vector_bucket(s3vectors, bucket_name: str = VECTOR_BUCKET_NAME):
    """
    Create a vector bucket.

    Vector buckets are specialized S3 buckets for storing vector indexes.
    Name must be 3-63 characters, lowercase letters/numbers/hyphens.
    """
    try:
        response = s3vectors.create_vector_bucket(
            vectorBucketName=bucket_name
        )
        print(f"Created vector bucket: {bucket_name}")
        return response
    except s3vectors.exceptions.ConflictException:
        print(f"Vector bucket already exists: {bucket_name}")
        return None


def list_vector_buckets(s3vectors) -> list[dict]:
    """List all vector buckets in the account."""
    response = s3vectors.list_vector_buckets()
    buckets = response.get("vectorBuckets", [])
    print(f"Found {len(buckets)} vector buckets")
    return buckets


def delete_vector_bucket(s3vectors, bucket_name: str):
    """Delete a vector bucket (must be empty)."""
    s3vectors.delete_vector_bucket(vectorBucketName=bucket_name)
    print(f"Deleted vector bucket: {bucket_name}")


# =============================================================================
# Vector Index Operations
# =============================================================================

def create_index(
    s3vectors,
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME,
    dimension: int = EMBEDDING_DIM,
    distance_metric: str = "cosine"
):
    """
    Create a vector index within a bucket.

    Parameters:
    - dimension: 1-4096, must match your embedding model output
    - distance_metric: "cosine" or "euclidean"
    - non_filterable_metadata_keys: Metadata keys excluded from filtering
      (useful for storing large text that shouldn't be indexed)

    IMPORTANT: These settings cannot be changed after creation!
    """
    try:
        response = s3vectors.create_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dimension=dimension,
            distanceMetric=distance_metric,
            metadataConfiguration={
                # Keys that won't be used in filter queries
                # Good for storing source text, descriptions, etc.
                "nonFilterableMetadataKeys": ["source_text", "full_content"]
            }
        )
        print(f"Created index: {index_name} (dim={dimension}, metric={distance_metric})")
        return response
    except s3vectors.exceptions.ConflictException:
        print(f"Index already exists: {index_name}")
        return None


def list_indexes(s3vectors, bucket_name: str = VECTOR_BUCKET_NAME) -> list[dict]:
    """List all indexes in a vector bucket."""
    response = s3vectors.list_indexes(vectorBucketName=bucket_name)
    indexes = response.get("indexes", [])
    print(f"Found {len(indexes)} indexes in {bucket_name}")
    return indexes


def get_index(s3vectors, bucket_name: str, index_name: str) -> dict:
    """Get details about a specific index."""
    response = s3vectors.get_index(
        vectorBucketName=bucket_name,
        indexName=index_name
    )
    return response


def delete_index(s3vectors, bucket_name: str, index_name: str):
    """Delete a vector index."""
    s3vectors.delete_index(
        vectorBucketName=bucket_name,
        indexName=index_name
    )
    print(f"Deleted index: {index_name}")


# =============================================================================
# Vector Operations
# =============================================================================

def put_vectors(
    s3vectors,
    embedder,
    documents: list[dict],
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME
):
    """
    Insert vectors with metadata into an index.

    Each document should have:
    - key: Unique identifier (string)
    - text: Text to embed
    - metadata: Dict of filterable/non-filterable fields
    """
    # Prepare vectors
    vectors = []
    for doc in documents:
        # Generate embedding
        text_to_embed = f"{doc.get('title', '')} {doc.get('content', '')}"
        embedding = embedder.embed_single(text_to_embed)

        vector = {
            "key": doc["key"],
            "data": {"float32": embedding},
            "metadata": {
                # Filterable metadata
                "title": doc.get("title", ""),
                "category": doc.get("category", "general"),
                # Non-filterable (for retrieval only)
                "source_text": doc.get("content", "")[:500]  # Truncate for storage
            }
        }
        vectors.append(vector)

    # Batch insert
    response = s3vectors.put_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        vectors=vectors
    )

    print(f"Inserted {len(vectors)} vectors")
    return response


def get_vectors(
    s3vectors,
    keys: list[str],
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME
) -> list[dict]:
    """Retrieve specific vectors by their keys."""
    response = s3vectors.get_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        keys=keys,
        returnMetadata=True
    )
    return response.get("vectors", [])


def delete_vectors(
    s3vectors,
    keys: list[str],
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME
):
    """Delete specific vectors by their keys."""
    response = s3vectors.delete_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        keys=keys
    )
    print(f"Deleted {len(keys)} vectors")
    return response


# =============================================================================
# Query Operations
# =============================================================================

def query_vectors(
    s3vectors,
    embedder,
    query_text: str,
    top_k: int = 10,
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME,
    return_metadata: bool = True,
    return_distance: bool = True
) -> list[dict]:
    """
    Perform similarity search.

    The query vector must have the same dimension as the index.
    """
    # Generate query embedding
    query_embedding = embedder.embed_single(query_text)

    response = s3vectors.query_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        queryVector={"float32": query_embedding},
        topK=top_k,
        returnMetadata=return_metadata,
        returnDistance=return_distance
    )

    return response.get("vectors", [])


def filtered_query(
    s3vectors,
    embedder,
    query_text: str,
    filter_dict: dict,
    top_k: int = 10,
    bucket_name: str = VECTOR_BUCKET_NAME,
    index_name: str = INDEX_NAME
) -> list[dict]:
    """
    Similarity search with metadata filter.

    Filter applies to filterable metadata keys only.
    Non-filterable keys (specified at index creation) cannot be used in filters.

    Example filters:
    - {"category": "tech"}
    - {"category": {"$in": ["tech", "science"]}}
    """
    query_embedding = embedder.embed_single(query_text)

    response = s3vectors.query_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        queryVector={"float32": query_embedding},
        topK=top_k,
        filter=filter_dict,
        returnMetadata=True,
        returnDistance=True
    )

    return response.get("vectors", [])


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "key": "doc_001",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
        "category": "tech"
    },
    {
        "key": "doc_002",
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses multi-layer neural networks to model complex patterns. Convolutional neural networks excel at image recognition while transformers dominate natural language processing.",
        "category": "tech"
    },
    {
        "key": "doc_003",
        "title": "Vector Databases Explained",
        "content": "Vector databases store high-dimensional embeddings and enable similarity search. Popular algorithms include HNSW and IVF for approximate nearest neighbor search.",
        "category": "tech"
    },
    {
        "key": "doc_004",
        "title": "The Art of Italian Cooking",
        "content": "Italian cuisine emphasizes fresh ingredients and simple preparations. From pasta to risotto, the key is using quality olive oil, fresh herbs, and ripe tomatoes.",
        "category": "food"
    },
    {
        "key": "doc_005",
        "title": "Mediterranean Diet Benefits",
        "content": "The Mediterranean diet focuses on vegetables, fruits, whole grains, and healthy fats. Studies show it reduces heart disease risk and promotes longevity.",
        "category": "food"
    },
    {
        "key": "doc_006",
        "title": "Cloud Computing Fundamentals",
        "content": "Cloud computing provides on-demand computing resources over the internet. Major providers include AWS, Azure, and Google Cloud Platform.",
        "category": "tech"
    },
    {
        "key": "doc_007",
        "title": "Sustainable Agriculture Practices",
        "content": "Sustainable farming minimizes environmental impact through crop rotation, organic methods, and water conservation. It ensures long-term food security.",
        "category": "environment"
    },
    {
        "key": "doc_008",
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand human language. Applications include sentiment analysis, machine translation, and chatbots powered by large language models.",
        "category": "tech"
    }
]


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Run the complete S3 Vectors example."""

    print("=" * 60)
    print("Amazon S3 Vectors Example")
    print("=" * 60)

    # Initialize clients
    s3vectors, bedrock = get_clients()

    # Choose embedder
    print("\n--- Initializing Embedder ---")
    try:
        # Try Bedrock first (production)
        embedder = BedrockEmbedder(bedrock)
        print("Using Amazon Bedrock Titan embeddings")
    except Exception as e:
        print(f"Bedrock not available ({e}), using local embeddings")
        embedder = LocalEmbedder()

    # Create vector bucket
    print("\n--- Creating Vector Bucket ---")
    create_vector_bucket(s3vectors)

    # List buckets
    buckets = list_vector_buckets(s3vectors)
    for b in buckets:
        print(f"  - {b.get('vectorBucketName')}")

    # Create index
    print("\n--- Creating Vector Index ---")
    create_index(
        s3vectors,
        dimension=EMBEDDING_DIM,
        distance_metric="cosine"
    )

    # List indexes
    indexes = list_indexes(s3vectors)
    for idx in indexes:
        print(f"  - {idx.get('indexName')}")

    # Insert vectors
    print("\n--- Inserting Vectors ---")
    put_vectors(s3vectors, embedder, SAMPLE_DOCUMENTS)

    # Wait for indexing (S3 Vectors may need a moment)
    print("Waiting for index to update...")
    time.sleep(2)

    # Basic query
    print("\n--- Query: 'machine learning algorithms' ---")
    results = query_vectors(s3vectors, embedder, "machine learning algorithms", top_k=3)
    for r in results:
        distance = r.get("distance", "N/A")
        metadata = r.get("metadata", {})
        print(f"  [{distance:.4f}] {metadata.get('title', r['key'])}")

    # Filtered query
    print("\n--- Filtered Query: 'healthy eating' (category=food) ---")
    results = filtered_query(
        s3vectors, embedder,
        "healthy eating",
        filter_dict={"category": "food"},
        top_k=3
    )
    for r in results:
        distance = r.get("distance", "N/A")
        metadata = r.get("metadata", {})
        print(f"  [{distance:.4f}] {metadata.get('title', r['key'])}")

    # Another query
    print("\n--- Query: 'cloud infrastructure' ---")
    results = query_vectors(s3vectors, embedder, "cloud infrastructure", top_k=3)
    for r in results:
        distance = r.get("distance", "N/A")
        metadata = r.get("metadata", {})
        print(f"  [{distance:.4f}] {metadata.get('title', r['key'])}")

    # Retrieve specific vectors
    print("\n--- Retrieving Specific Vectors ---")
    vectors = get_vectors(s3vectors, ["doc_001", "doc_002"])
    for v in vectors:
        print(f"  Key: {v['key']}, Has embedding: {'data' in v}")

    # Get index details
    print("\n--- Index Details ---")
    index_info = get_index(s3vectors, VECTOR_BUCKET_NAME, INDEX_NAME)
    print(f"  Dimension: {index_info.get('dimension')}")
    print(f"  Distance Metric: {index_info.get('distanceMetric')}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    # Cleanup prompt
    print("\nTo clean up resources, uncomment the following in the code:")
    print("  delete_index(s3vectors, VECTOR_BUCKET_NAME, INDEX_NAME)")
    print("  delete_vector_bucket(s3vectors, VECTOR_BUCKET_NAME)")


def cleanup():
    """Clean up all created resources."""
    s3vectors, _ = get_clients()

    try:
        # Delete all vectors first (or delete the index directly)
        delete_index(s3vectors, VECTOR_BUCKET_NAME, INDEX_NAME)
    except Exception as e:
        print(f"Could not delete index: {e}")

    try:
        delete_vector_bucket(s3vectors, VECTOR_BUCKET_NAME)
    except Exception as e:
        print(f"Could not delete bucket: {e}")


if __name__ == "__main__":
    main()
    # Uncomment to clean up:
    # cleanup()
