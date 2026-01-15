"""
OpenSearch k-NN Vector Search Example

This example demonstrates:
1. Creating an index with HNSW configuration
2. Bulk indexing documents with embeddings
3. Performing similarity search
4. Hybrid search (BM25 + vector)
5. Filtered vector search

Requirements:
    pip install opensearch-py sentence-transformers numpy

For AWS OpenSearch Service, also:
    pip install boto3 requests-aws4auth
"""

import json
import numpy as np
from typing import Optional
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

OPENSEARCH_CONFIG = {
    # Local OpenSearch
    "host": "localhost",
    "port": 9200,
    "use_ssl": False,
    "http_auth": ("admin", "admin"),  # Default for local dev

    # For AWS OpenSearch Service, uncomment and configure:
    # "host": "your-domain.us-east-1.es.amazonaws.com",
    # "port": 443,
    # "use_ssl": True,
    # "http_auth": get_aws_auth(),  # See helper function below
}

INDEX_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# =============================================================================
# AWS Authentication Helper (for OpenSearch Service)
# =============================================================================

def get_aws_auth():
    """Get AWS authentication for OpenSearch Service."""
    import boto3
    from requests_aws4auth import AWS4Auth

    credentials = boto3.Session().get_credentials()
    region = "us-east-1"  # Change to your region

    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token
    )


# =============================================================================
# OpenSearch Client Setup
# =============================================================================

def create_client() -> OpenSearch:
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_CONFIG["host"], "port": OPENSEARCH_CONFIG["port"]}],
        http_auth=OPENSEARCH_CONFIG.get("http_auth"),
        use_ssl=OPENSEARCH_CONFIG.get("use_ssl", False),
        verify_certs=OPENSEARCH_CONFIG.get("verify_certs", False),
        ssl_show_warn=False
    )


# =============================================================================
# Index Management
# =============================================================================

def create_index(client: OpenSearch, index_name: str = INDEX_NAME):
    """
    Create an index with HNSW vector configuration.

    Key parameters explained:
    - m: Max edges per node (16-128). Higher = better recall, more memory
    - ef_construction: Build-time candidate pool (100-500). Higher = better graph
    - ef_search: Query-time candidate pool. Set via index settings
    """

    # Delete if exists
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")

    # Index configuration with HNSW
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,  # Query-time parameter
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",  # cosine similarity
                        "engine": "faiss",  # or "nmslib", "lucene"
                        "parameters": {
                            "ef_construction": 256,  # Build-time quality
                            "m": 32  # Graph connectivity
                        }
                    }
                },
                "title": {"type": "text"},
                "content": {"type": "text"},
                "category": {"type": "keyword"},
                "created_at": {"type": "date"}
            }
        }
    }

    response = client.indices.create(index=index_name, body=index_body)
    print(f"Created index: {index_name}")
    return response


def create_ivf_index(client: OpenSearch, index_name: str = "documents_ivf"):
    """
    Alternative: Create index with IVF configuration.

    IVF is better for:
    - Memory-constrained environments
    - Larger datasets where HNSW memory is prohibitive
    """

    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "ivf",
                        "space_type": "l2",
                        "engine": "faiss",
                        "parameters": {
                            "nlist": 128,  # Number of clusters
                            "nprobes": 8   # Clusters to search
                        }
                    }
                },
                "title": {"type": "text"},
                "content": {"type": "text"}
            }
        }
    }

    response = client.indices.create(index=index_name, body=index_body)
    print(f"Created IVF index: {index_name}")
    return response


# =============================================================================
# Embedding Generation
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model dimension: {self.dimension}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, normalize_embeddings=True).tolist()


# =============================================================================
# Document Indexing
# =============================================================================

def index_documents(
    client: OpenSearch,
    embedder: EmbeddingGenerator,
    documents: list[dict],
    index_name: str = INDEX_NAME
):
    """
    Bulk index documents with embeddings.

    Each document should have: title, content, category (optional)
    """

    # Generate embeddings for all documents
    texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    embeddings = embedder.embed(texts)

    # Prepare bulk actions
    actions = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        action = {
            "_index": index_name,
            "_id": doc.get("id", str(i)),
            "_source": {
                "title": doc["title"],
                "content": doc["content"],
                "category": doc.get("category", "general"),
                "embedding": embedding.tolist()
            }
        }
        actions.append(action)

    # Bulk index
    success, errors = helpers.bulk(client, actions)
    print(f"Indexed {success} documents, {len(errors)} errors")

    # Refresh index
    client.indices.refresh(index=index_name)
    return success


# =============================================================================
# Search Operations
# =============================================================================

def vector_search(
    client: OpenSearch,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    index_name: str = INDEX_NAME
) -> list[dict]:
    """
    Perform pure vector similarity search.
    """

    # Generate query embedding
    query_embedding = embedder.embed_single(query)

    # k-NN query
    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        },
        "_source": ["title", "content", "category"]
    }

    response = client.search(index=index_name, body=search_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"],
            "category": hit["_source"]["category"]
        })

    return results


def filtered_vector_search(
    client: OpenSearch,
    embedder: EmbeddingGenerator,
    query: str,
    category: str,
    k: int = 10,
    index_name: str = INDEX_NAME
) -> list[dict]:
    """
    Vector search with metadata filter (pre-filtering).
    """

    query_embedding = embedder.embed_single(query)

    # k-NN with filter
    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k,
                    "filter": {
                        "term": {"category": category}
                    }
                }
            }
        },
        "_source": ["title", "content", "category"]
    }

    response = client.search(index=index_name, body=search_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"]
        })

    return results


def hybrid_search(
    client: OpenSearch,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    index_name: str = INDEX_NAME
) -> list[dict]:
    """
    Hybrid search combining BM25 text search with vector similarity.

    Note: Requires OpenSearch 2.10+ with search pipelines configured,
    or manual score combination.
    """

    query_embedding = embedder.embed_single(query)

    # Manual hybrid: run both queries and combine
    # Option 1: Bool query with function_score
    search_body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    # BM25 text match
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "content"],
                            "boost": 0.3
                        }
                    },
                    # Vector similarity
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k * 2,  # Fetch more for reranking
                                "boost": 0.7
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["title", "content", "category"]
    }

    response = client.search(index=index_name, body=search_body)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"]
        })

    return results


# =============================================================================
# Tuning Helpers
# =============================================================================

def update_ef_search(client: OpenSearch, ef_search: int, index_name: str = INDEX_NAME):
    """
    Update ef_search parameter without reindexing.
    Higher values = better recall, slower queries.
    """
    client.indices.put_settings(
        index=index_name,
        body={"index": {"knn.algo_param.ef_search": ef_search}}
    )
    print(f"Updated ef_search to {ef_search}")


def get_index_stats(client: OpenSearch, index_name: str = INDEX_NAME) -> dict:
    """Get index statistics including memory usage."""
    stats = client.indices.stats(index=index_name)

    return {
        "doc_count": stats["indices"][index_name]["primaries"]["docs"]["count"],
        "size_bytes": stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"],
        "size_mb": stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"] / (1024 * 1024)
    }


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
        "category": "tech"
    },
    {
        "id": "2",
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses multi-layer neural networks to model complex patterns. Convolutional neural networks excel at image recognition while transformers dominate natural language processing.",
        "category": "tech"
    },
    {
        "id": "3",
        "title": "Vector Databases Explained",
        "content": "Vector databases store high-dimensional embeddings and enable similarity search. Popular algorithms include HNSW and IVF for approximate nearest neighbor search.",
        "category": "tech"
    },
    {
        "id": "4",
        "title": "The Art of Italian Cooking",
        "content": "Italian cuisine emphasizes fresh ingredients and simple preparations. From pasta to risotto, the key is using quality olive oil, fresh herbs, and ripe tomatoes.",
        "category": "food"
    },
    {
        "id": "5",
        "title": "Mediterranean Diet Benefits",
        "content": "The Mediterranean diet focuses on vegetables, fruits, whole grains, and healthy fats. Studies show it reduces heart disease risk and promotes longevity.",
        "category": "food"
    },
    {
        "id": "6",
        "title": "Cloud Computing Fundamentals",
        "content": "Cloud computing provides on-demand computing resources over the internet. Major providers include AWS, Azure, and Google Cloud Platform.",
        "category": "tech"
    },
    {
        "id": "7",
        "title": "Sustainable Agriculture Practices",
        "content": "Sustainable farming minimizes environmental impact through crop rotation, organic methods, and water conservation. It ensures long-term food security.",
        "category": "environment"
    },
    {
        "id": "8",
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand human language. Applications include sentiment analysis, machine translation, and chatbots powered by large language models.",
        "category": "tech"
    }
]


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Run the complete OpenSearch example."""

    print("=" * 60)
    print("OpenSearch k-NN Vector Search Example")
    print("=" * 60)

    # Initialize
    client = create_client()
    embedder = EmbeddingGenerator()

    # Check connection
    info = client.info()
    print(f"\nConnected to OpenSearch {info['version']['number']}")

    # Create index
    print("\n--- Creating Index ---")
    create_index(client)

    # Index documents
    print("\n--- Indexing Documents ---")
    index_documents(client, embedder, SAMPLE_DOCUMENTS)

    # Get stats
    stats = get_index_stats(client)
    print(f"Index stats: {stats['doc_count']} docs, {stats['size_mb']:.2f} MB")

    # Vector search
    print("\n--- Vector Search: 'machine learning algorithms' ---")
    results = vector_search(client, embedder, "machine learning algorithms", k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['title']}")

    # Filtered search
    print("\n--- Filtered Search: 'healthy eating' (category=food) ---")
    results = filtered_vector_search(
        client, embedder, "healthy eating", category="food", k=3
    )
    for r in results:
        print(f"  [{r['score']:.4f}] {r['title']}")

    # Hybrid search
    print("\n--- Hybrid Search: 'neural networks for images' ---")
    results = hybrid_search(client, embedder, "neural networks for images", k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['title']}")

    # Demonstrate parameter tuning
    print("\n--- Parameter Tuning ---")
    print("Updating ef_search from 100 to 200...")
    update_ef_search(client, 200)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
