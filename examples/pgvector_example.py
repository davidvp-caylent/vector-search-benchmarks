"""
pgvector Vector Search Example (PostgreSQL)

This example demonstrates:
1. Setting up pgvector extension
2. Creating tables with vector columns
3. Building HNSW and IVFFlat indexes
4. Performing similarity search with different distance metrics
5. Combining vector search with SQL filters
6. Tuning index parameters

Requirements:
    pip install psycopg2-binary sentence-transformers numpy

PostgreSQL Setup:
    CREATE EXTENSION vector;
"""

import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vectordb",
    "user": "postgres",
    "password": "postgres"
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# =============================================================================
# Database Setup
# =============================================================================

def get_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)


def setup_database(conn):
    """
    Initialize pgvector extension and create tables.
    """
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop existing table
        cur.execute("DROP TABLE IF EXISTS documents CASCADE;")

        # Create table with vector column
        cur.execute(f"""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                embedding vector({EMBEDDING_DIM}),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        conn.commit()
        print("Database setup complete")


def create_hnsw_index(conn, m: int = 16, ef_construction: int = 128):
    """
    Create HNSW index for fast similarity search.

    Parameters:
    - m: Max connections per layer (default 16, range 2-100)
      Higher = better recall, more memory
    - ef_construction: Build-time candidate list size (default 64)
      Higher = better index quality, slower build

    Distance operators:
    - vector_cosine_ops: Cosine distance (<=>)
    - vector_l2_ops: Euclidean distance (<->)
    - vector_ip_ops: Inner product (<#>)
    """
    with conn.cursor() as cur:
        # Drop existing index
        cur.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx;")

        # Create HNSW index
        cur.execute(f"""
            CREATE INDEX documents_embedding_hnsw_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """)

        conn.commit()
        print(f"Created HNSW index (m={m}, ef_construction={ef_construction})")


def create_ivfflat_index(conn, lists: int = 100):
    """
    Create IVFFlat index - faster to build, less memory than HNSW.

    Parameters:
    - lists: Number of clusters/partitions
      Recommended: rows/1000 for <1M rows, sqrt(rows) for >1M rows

    Note: IVFFlat requires data in the table before creating the index!
    """
    with conn.cursor() as cur:
        # Check row count
        cur.execute("SELECT COUNT(*) FROM documents;")
        row_count = cur.fetchone()[0]

        if row_count == 0:
            print("WARNING: IVFFlat requires data before index creation!")
            return

        # Drop existing index
        cur.execute("DROP INDEX IF EXISTS documents_embedding_ivfflat_idx;")

        # Create IVFFlat index
        cur.execute(f"""
            CREATE INDEX documents_embedding_ivfflat_idx
            ON documents
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)

        conn.commit()
        print(f"Created IVFFlat index (lists={lists})")


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
# Document Operations
# =============================================================================

def insert_documents(conn, embedder: EmbeddingGenerator, documents: list[dict]):
    """
    Bulk insert documents with embeddings.
    """
    # Generate embeddings
    texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    embeddings = embedder.embed(texts)

    # Prepare data
    data = []
    for doc, emb in zip(documents, embeddings):
        data.append((
            doc["title"],
            doc["content"],
            doc.get("category", "general"),
            emb.tolist()
        ))

    # Bulk insert
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO documents (title, content, category, embedding)
            VALUES %s
            """,
            data,
            template="(%s, %s, %s, %s::vector)"
        )
        conn.commit()

    print(f"Inserted {len(documents)} documents")


# =============================================================================
# Search Operations
# =============================================================================

def cosine_search(
    conn,
    embedder: EmbeddingGenerator,
    query: str,
    limit: int = 10,
    ef_search: int = 100
) -> list[dict]:
    """
    Cosine similarity search using <=> operator.

    Lower distance = more similar (cosine distance = 1 - cosine_similarity)
    """
    query_embedding = embedder.embed_single(query)

    with conn.cursor() as cur:
        # Set HNSW search parameter (per-session)
        cur.execute(f"SET hnsw.ef_search = {ef_search};")

        cur.execute("""
            SELECT
                id,
                title,
                content,
                category,
                embedding <=> %s::vector AS distance
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, limit))

        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "category": row[3],
                "distance": row[4],
                "similarity": 1 - row[4]  # Convert distance to similarity
            })

        return results


def euclidean_search(
    conn,
    embedder: EmbeddingGenerator,
    query: str,
    limit: int = 10
) -> list[dict]:
    """
    Euclidean (L2) distance search using <-> operator.

    Note: Requires index with vector_l2_ops operator class.
    """
    query_embedding = embedder.embed_single(query)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                title,
                content,
                embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, limit))

        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "distance": row[3]
            })

        return results


def filtered_search(
    conn,
    embedder: EmbeddingGenerator,
    query: str,
    category: str,
    limit: int = 10
) -> list[dict]:
    """
    Vector search with SQL WHERE filter.

    Note: For pgvector 0.8+, enable iterative scans for better filtered results:
    SET hnsw.iterative_scan = relaxed_order;
    """
    query_embedding = embedder.embed_single(query)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                title,
                content,
                category,
                embedding <=> %s::vector AS distance
            FROM documents
            WHERE category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, category, query_embedding, limit))

        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "category": row[3],
                "distance": row[4]
            })

        return results


def hybrid_search(
    conn,
    embedder: EmbeddingGenerator,
    query: str,
    limit: int = 10,
    text_weight: float = 0.3,
    vector_weight: float = 0.7
) -> list[dict]:
    """
    Hybrid search combining full-text search with vector similarity.

    Uses PostgreSQL's built-in full-text search (tsvector/tsquery)
    combined with vector similarity scores.
    """
    query_embedding = embedder.embed_single(query)

    with conn.cursor() as cur:
        # Hybrid scoring: combine text relevance with vector similarity
        cur.execute("""
            SELECT
                id,
                title,
                content,
                category,
                -- Text relevance score (0-1 normalized)
                COALESCE(
                    ts_rank(
                        to_tsvector('english', title || ' ' || content),
                        plainto_tsquery('english', %s)
                    ),
                    0
                ) AS text_score,
                -- Vector similarity (convert distance to similarity)
                1 - (embedding <=> %s::vector) AS vector_score
            FROM documents
            ORDER BY (
                %s * COALESCE(
                    ts_rank(
                        to_tsvector('english', title || ' ' || content),
                        plainto_tsquery('english', %s)
                    ),
                    0
                ) +
                %s * (1 - (embedding <=> %s::vector))
            ) DESC
            LIMIT %s;
        """, (
            query, query_embedding,
            text_weight, query,
            vector_weight, query_embedding,
            limit
        ))

        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "category": row[3],
                "text_score": row[4],
                "vector_score": row[5],
                "combined_score": text_weight * row[4] + vector_weight * row[5]
            })

        return results


# =============================================================================
# Index Tuning
# =============================================================================

def set_ivfflat_probes(conn, probes: int):
    """
    Set number of clusters to search for IVFFlat.

    Higher probes = better recall, slower queries.
    Recommended starting point: sqrt(lists)
    """
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {probes};")
        print(f"Set ivfflat.probes = {probes}")


def set_hnsw_ef_search(conn, ef_search: int):
    """
    Set HNSW search candidate pool size.

    Higher ef_search = better recall, slower queries.
    Must be >= k (number of results requested).
    """
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        print(f"Set hnsw.ef_search = {ef_search}")


def enable_iterative_scan(conn):
    """
    Enable iterative index scans (pgvector 0.8+).

    Helps prevent over-filtering when combining vector search with WHERE clauses.
    """
    with conn.cursor() as cur:
        cur.execute("SET hnsw.iterative_scan = relaxed_order;")
        cur.execute("SET hnsw.max_scan_tuples = 20000;")
        print("Enabled iterative scans for HNSW")


def get_index_info(conn) -> dict:
    """Get information about existing indexes."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = 'documents';
        """)

        indexes = {}
        for row in cur.fetchall():
            indexes[row[0]] = row[1]

        # Get table size
        cur.execute("""
            SELECT
                pg_size_pretty(pg_total_relation_size('documents')) as total_size,
                (SELECT COUNT(*) FROM documents) as row_count;
        """)
        size_info = cur.fetchone()

        return {
            "indexes": indexes,
            "total_size": size_info[0],
            "row_count": size_info[1]
        }


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
        "category": "tech"
    },
    {
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses multi-layer neural networks to model complex patterns. Convolutional neural networks excel at image recognition while transformers dominate natural language processing.",
        "category": "tech"
    },
    {
        "title": "Vector Databases Explained",
        "content": "Vector databases store high-dimensional embeddings and enable similarity search. Popular algorithms include HNSW and IVF for approximate nearest neighbor search.",
        "category": "tech"
    },
    {
        "title": "The Art of Italian Cooking",
        "content": "Italian cuisine emphasizes fresh ingredients and simple preparations. From pasta to risotto, the key is using quality olive oil, fresh herbs, and ripe tomatoes.",
        "category": "food"
    },
    {
        "title": "Mediterranean Diet Benefits",
        "content": "The Mediterranean diet focuses on vegetables, fruits, whole grains, and healthy fats. Studies show it reduces heart disease risk and promotes longevity.",
        "category": "food"
    },
    {
        "title": "Cloud Computing Fundamentals",
        "content": "Cloud computing provides on-demand computing resources over the internet. Major providers include AWS, Azure, and Google Cloud Platform.",
        "category": "tech"
    },
    {
        "title": "Sustainable Agriculture Practices",
        "content": "Sustainable farming minimizes environmental impact through crop rotation, organic methods, and water conservation. It ensures long-term food security.",
        "category": "environment"
    },
    {
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand human language. Applications include sentiment analysis, machine translation, and chatbots powered by large language models.",
        "category": "tech"
    }
]


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Run the complete pgvector example."""

    print("=" * 60)
    print("pgvector Vector Search Example")
    print("=" * 60)

    # Initialize
    conn = get_connection()
    embedder = EmbeddingGenerator()

    # Setup database
    print("\n--- Database Setup ---")
    setup_database(conn)

    # Insert documents
    print("\n--- Inserting Documents ---")
    insert_documents(conn, embedder, SAMPLE_DOCUMENTS)

    # Create HNSW index
    print("\n--- Creating HNSW Index ---")
    create_hnsw_index(conn, m=16, ef_construction=128)

    # Get index info
    info = get_index_info(conn)
    print(f"Table size: {info['total_size']}, Rows: {info['row_count']}")
    print(f"Indexes: {list(info['indexes'].keys())}")

    # Cosine similarity search
    print("\n--- Cosine Search: 'machine learning algorithms' ---")
    results = cosine_search(conn, embedder, "machine learning algorithms", limit=3)
    for r in results:
        print(f"  [sim={r['similarity']:.4f}] {r['title']}")

    # Filtered search
    print("\n--- Filtered Search: 'healthy eating' (category=food) ---")
    results = filtered_search(conn, embedder, "healthy eating", category="food", limit=3)
    for r in results:
        print(f"  [dist={r['distance']:.4f}] {r['title']}")

    # Hybrid search
    print("\n--- Hybrid Search: 'neural networks for images' ---")
    results = hybrid_search(conn, embedder, "neural networks for images", limit=3)
    for r in results:
        print(f"  [combined={r['combined_score']:.4f}] {r['title']}")
        print(f"      text={r['text_score']:.4f}, vector={r['vector_score']:.4f}")

    # Demonstrate IVFFlat
    print("\n--- Creating IVFFlat Index ---")
    create_ivfflat_index(conn, lists=4)  # Small for demo
    set_ivfflat_probes(conn, 2)

    # Compare tuning
    print("\n--- Parameter Tuning Comparison ---")
    print("Testing different ef_search values...")
    for ef in [10, 50, 100, 200]:
        set_hnsw_ef_search(conn, ef)
        results = cosine_search(conn, embedder, "machine learning", limit=3, ef_search=ef)
        print(f"  ef_search={ef}: top result = {results[0]['title']}")

    conn.close()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
