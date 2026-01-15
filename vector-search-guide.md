# Vector Search, Embeddings, and Indexing: A Practical Guide

A comprehensive guide to vector search across **OpenSearch**, **pgvector**, and **Amazon S3 Vectors**, including working examples and benchmarking strategies.

---

## Table of Contents

1. [Vector Search Fundamentals](#vector-search-fundamentals)
2. [Understanding Embeddings](#understanding-embeddings)
3. [Distance Metrics](#distance-metrics)
4. [Indexing Algorithms Deep Dive](#indexing-algorithms-deep-dive)
5. [Platform Comparisons](#platform-comparisons)
6. [OpenSearch k-NN](#opensearch-k-nn)
7. [pgvector (PostgreSQL)](#pgvector-postgresql)
8. [Amazon S3 Vectors](#amazon-s3-vectors)
9. [Precision and Recall Experiments](#precision-and-recall-experiments)
10. [When to Use What](#when-to-use-what)

---

## Vector Search Fundamentals

Vector search (also called similarity search or semantic search) finds items that are "similar" to a query based on their vector representations rather than exact keyword matches.

### How It Works

```
Traditional Search:  "red shoes" → matches documents containing "red" AND "shoes"
Vector Search:       "red shoes" → matches documents semantically similar (sneakers, crimson boots, etc.)
```

### The Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │ ──► │  Embedding  │ ──► │   Vector    │ ──► │   Index     │
│ (text/img)  │     │   Model     │     │  [0.1, 0.3] │     │  (ANN/KNN)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│   Results   │ ◄── │  Distance   │ ◄── │   Query     │ ◄──────────┘
│  (top-k)    │     │  Ranking    │     │   Vector    │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Understanding Embeddings

Embeddings are dense vector representations that capture semantic meaning in a continuous vector space.

### What Makes a Good Embedding?

- **Semantic similarity**: Similar concepts should be close in vector space
- **Dimensionality**: Higher dimensions capture more nuance but cost more
- **Normalization**: Most models output normalized vectors (unit length)

### Common Embedding Dimensions

| Model | Dimensions | Use Case |
|-------|------------|----------|
| OpenAI text-embedding-3-small | 1536 | General purpose |
| OpenAI text-embedding-3-large | 3072 | High accuracy |
| Amazon Titan Text v2 | 1024 | AWS-native workloads |
| Cohere embed-v3 | 1024 | Multilingual |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Lightweight/fast |
| BAAI/bge-large-en-v1.5 | 1024 | Open source, high quality |

### Dimension Trade-offs

```
Lower Dimensions (384)          Higher Dimensions (1536+)
├─ Faster indexing              ├─ Better semantic capture
├─ Less memory                  ├─ Higher accuracy
├─ Lower storage costs          ├─ More computational cost
└─ May lose nuance              └─ Slower queries (without optimization)
```

---

## Distance Metrics

The distance metric determines how "closeness" is calculated between vectors.

### Cosine Similarity

Measures the angle between vectors, ignoring magnitude. Best for normalized embeddings.

```
cosine_similarity = (A · B) / (||A|| × ||B||)

Range: -1 to 1 (1 = identical direction)
```

**When to use**: Text embeddings, when magnitude shouldn't matter

### Euclidean Distance (L2)

Measures straight-line distance between points.

```
euclidean = √(Σ(Aᵢ - Bᵢ)²)

Range: 0 to ∞ (0 = identical)
```

**When to use**: Image embeddings, spatial data, when magnitude matters

### Dot Product (Inner Product)

Measures both direction and magnitude alignment.

```
dot_product = Σ(Aᵢ × Bᵢ)

Range: -∞ to ∞ (higher = more similar)
```

**When to use**: Recommendation systems, when you want to factor in "strength"

### Metric Comparison

| Metric | Normalized Vectors | Speed | Best For |
|--------|-------------------|-------|----------|
| Cosine | Required for meaningful results | Fast | Text, semantic search |
| Euclidean | Not required | Moderate | Images, spatial data |
| Dot Product | Optional (changes interpretation) | Fastest | Recommendations |

---

## Indexing Algorithms Deep Dive

Exact k-NN search is O(n) - impractical at scale. Approximate Nearest Neighbor (ANN) algorithms trade some accuracy for massive speed gains.

### HNSW (Hierarchical Navigable Small World)

The most popular algorithm for high-recall, low-latency search.

#### How It Works

```
Layer 2 (sparse):    A ─────────────────── B
                     │                     │
Layer 1 (medium):    A ─── C ─── D ─── E ─ B
                     │     │     │     │   │
Layer 0 (dense):     A─F─G─C─H─I─D─J─K─E─L─B
```

- **Multi-layer graph**: Higher layers are sparser, enabling "highway" navigation
- **Greedy search**: Start at top, descend while getting closer to target
- **Construction**: Each new vector connects to M nearest neighbors per layer

#### Key Parameters

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `M` | Max edges per node per layer | Higher = better recall, more memory |
| `ef_construction` | Candidate pool during build | Higher = better graph, slower build |
| `ef_search` | Candidate pool during query | Higher = better recall, slower query |

#### Recommended Configurations (OpenSearch)

```python
# Config 1: Fast, lower recall
{"M": 16, "ef_construction": 128, "ef_search": 32}

# Config 2: Balanced
{"M": 32, "ef_construction": 128, "ef_search": 64}

# Config 3: High recall
{"M": 64, "ef_construction": 256, "ef_search": 128}

# Config 4: Maximum recall
{"M": 128, "ef_construction": 256, "ef_search": 256}
```

#### Memory Formula

```
Memory ≈ 1.1 × (4 × dimension + 8 × M) bytes per vector

Example: 1M vectors, 1024 dimensions, M=16
Memory ≈ 1.1 × (4 × 1024 + 8 × 16) × 1,000,000 = ~4.6 GB
```

### IVFFlat (Inverted File with Flat Quantization)

Partitions vectors into clusters, searches only relevant partitions.

#### How It Works

```
Training Phase:
┌─────────────────────────────────────────┐
│  Vectors clustered into N partitions    │
│  Each partition has a centroid          │
└─────────────────────────────────────────┘

Query Phase:
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Centroid │     │ Centroid │     │ Centroid │
│    A     │     │    B     │     │    C     │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
  [vectors]       [vectors]        [vectors]

Query → Find nearest centroids → Search those partitions only
```

#### Key Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `lists` | Number of partitions | rows/1000 (up to 1M), √rows (over 1M) |
| `probes` | Partitions to search at query time | √lists as starting point |

#### Trade-offs vs HNSW

| Aspect | IVFFlat | HNSW |
|--------|---------|------|
| Build time | Fast | Slow (32x slower) |
| Memory | Lower (2.8x less) | Higher |
| Query speed | Slower | Faster (15x faster) |
| Recall at speed | Lower | Higher |
| Can build empty? | No (needs data) | Yes |

### IVFPQ (IVF with Product Quantization)

Adds compression to IVF for memory-constrained scenarios.

#### How It Works

```
Original Vector: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Split into subvectors:
[0.1, 0.2] [0.3, 0.4] [0.5, 0.6] [0.7, 0.8]

Each subvector → Codebook ID (1 byte each)
[42] [17] [89] [3]

Compressed: 8 floats (32 bytes) → 4 bytes
```

#### Trade-offs

- **Pro**: 8-32x memory reduction
- **Con**: Lossy compression, lower recall
- **Use when**: Memory is critical constraint

---

## Platform Comparisons

| Feature | OpenSearch | pgvector | S3 Vectors |
|---------|------------|----------|------------|
| **Type** | Search engine | PostgreSQL extension | Object storage |
| **Max vectors** | Billions | Millions (practical) | 2B per index |
| **Indexes** | HNSW, IVF, IVFPQ | HNSW, IVFFlat | Managed (opaque) |
| **Query latency** | <10ms (warm) | 10-100ms | 100ms (warm) |
| **Filtering** | Advanced (BM25 + vector) | SQL WHERE | Metadata filters |
| **Best for** | High QPS, hybrid search | OLTP + vectors | Cost-optimized, RAG |
| **Managed option** | OpenSearch Service | RDS, Aurora | Native S3 |

---

## OpenSearch k-NN

OpenSearch provides native vector search through the k-NN plugin with multiple engines.

### Engines Comparison

| Engine | Algorithms | Best For |
|--------|------------|----------|
| **nmslib** | HNSW | Low latency, high recall |
| **faiss** | HNSW, IVF, IVFPQ | Large scale, memory optimization |
| **Lucene** | HNSW | Smaller datasets, filtering |

### Index Mapping Example

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "faiss",
          "parameters": {
            "ef_construction": 256,
            "m": 32
          }
        }
      },
      "title": { "type": "text" },
      "category": { "type": "keyword" }
    }
  }
}
```

### Query Example

```json
{
  "size": 10,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, ...],
        "k": 10
      }
    }
  }
}
```

### Hybrid Search (BM25 + Vector)

```json
{
  "size": 10,
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "title": "machine learning"
          }
        },
        {
          "knn": {
            "embedding": {
              "vector": [0.1, 0.2, ...],
              "k": 50
            }
          }
        }
      ]
    }
  }
}
```

### Tuning Guidelines

1. **Start with Config 1**, measure recall
2. **Increase ef_search** first (no reindex needed)
3. **Increase M** if still insufficient (requires reindex)
4. **Use faiss fp16** for memory reduction with minimal quality loss

---

## pgvector (PostgreSQL)

pgvector brings vector similarity search to PostgreSQL with familiar SQL semantics.

### Installation

```sql
CREATE EXTENSION vector;
```

### Schema Design

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024),  -- dimension must match your model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Index Creation

#### HNSW Index (Recommended for most cases)

```sql
-- Create HNSW index
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Set search parameter (session-level)
SET hnsw.ef_search = 100;
```

#### IVFFlat Index (Faster build, less memory)

```sql
-- Requires data in table first!
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);  -- rows/1000 for < 1M rows

-- Set search parameter
SET ivfflat.probes = 32;  -- sqrt(lists)
```

### Distance Operators

| Operator | Distance Type | Index Ops Class |
|----------|---------------|-----------------|
| `<->` | Euclidean (L2) | `vector_l2_ops` |
| `<=>` | Cosine | `vector_cosine_ops` |
| `<#>` | Negative inner product | `vector_ip_ops` |

### Query Examples

```sql
-- Basic similarity search
SELECT id, content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- With metadata filter
SELECT id, content
FROM documents
WHERE metadata->>'category' = 'tech'
ORDER BY embedding <=> $1
LIMIT 10;

-- Hybrid: full-text + vector
SELECT id, content,
       ts_rank(to_tsvector(content), plainto_tsquery('machine learning')) AS text_score,
       1 - (embedding <=> $1) AS vector_score
FROM documents
WHERE to_tsvector(content) @@ plainto_tsquery('machine learning')
ORDER BY (0.5 * ts_rank(...) + 0.5 * (1 - (embedding <=> $1))) DESC
LIMIT 10;
```

### Iterative Scans (pgvector 0.8+)

Prevents over-filtering issues:

```sql
SET hnsw.iterative_scan = relaxed_order;
SET hnsw.max_scan_tuples = 10000;

-- Now filtered queries will continue searching if needed
SELECT * FROM documents
WHERE category = 'rare_category'
ORDER BY embedding <=> $1
LIMIT 10;
```

### Tuning Guidelines

| Scenario | lists | probes | m | ef_construction |
|----------|-------|--------|---|-----------------|
| < 100K rows | 100 | 10 | 16 | 64 |
| 100K - 1M rows | 1000 | 32 | 16 | 128 |
| 1M - 10M rows | 3162 (√n) | 56 | 32 | 200 |
| > 10M rows | Consider partitioning | - | 48 | 256 |

---

## Amazon S3 Vectors

S3 Vectors is purpose-built for cost-optimized vector storage in AI applications.

### Key Characteristics

- **Scale**: 2 billion vectors per index, 10,000 indexes per bucket
- **Latency**: ~100ms warm, sub-second cold
- **Cost**: Up to 90% less than traditional vector DBs
- **Best for**: Infrequent queries, RAG, AI agent memory

### Architecture

```
┌─────────────────────────────────────────┐
│           Vector Bucket                 │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Index 1   │  │   Index 2   │ ...  │
│  │ (products)  │  │  (reviews)  │      │
│  │             │  │             │      │
│  │ 2B vectors  │  │ 2B vectors  │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

### Creating Resources

```python
import boto3

s3vectors = boto3.client('s3vectors', region_name='us-west-2')

# Create vector bucket
s3vectors.create_vector_bucket(
    vectorBucketName='my-embeddings'
)

# Create vector index
s3vectors.create_index(
    vectorBucketName='my-embeddings',
    indexName='documents',
    dimension=1024,
    distanceMetric='cosine',  # or 'euclidean'
    metadataConfiguration={
        'nonFilterableMetadataKeys': ['source_text']  # won't be used in filters
    }
)
```

### Inserting Vectors

```python
# Batch insert (recommended)
s3vectors.put_vectors(
    vectorBucketName='my-embeddings',
    indexName='documents',
    vectors=[
        {
            'key': 'doc_001',
            'data': {'float32': embedding_1},
            'metadata': {
                'title': 'Introduction to ML',
                'category': 'tech',
                'source_text': 'Full document text...'  # non-filterable
            }
        },
        {
            'key': 'doc_002',
            'data': {'float32': embedding_2},
            'metadata': {
                'title': 'Deep Learning Basics',
                'category': 'tech'
            }
        }
    ]
)
```

### Querying Vectors

```python
# Basic query
response = s3vectors.query_vectors(
    vectorBucketName='my-embeddings',
    indexName='documents',
    queryVector={'float32': query_embedding},
    topK=10,
    returnDistance=True,
    returnMetadata=True
)

# With metadata filter
response = s3vectors.query_vectors(
    vectorBucketName='my-embeddings',
    indexName='documents',
    queryVector={'float32': query_embedding},
    topK=10,
    filter={'category': 'tech'},
    returnDistance=True,
    returnMetadata=True
)

# Process results
for vector in response['vectors']:
    print(f"Key: {vector['key']}")
    print(f"Distance: {vector['distance']}")
    print(f"Metadata: {vector['metadata']}")
```

### Limitations & Considerations

| Aspect | Details |
|--------|---------|
| **Immutable settings** | Dimension, distance metric, index name cannot change |
| **Max dimension** | 4096 |
| **Latency** | Not for <10ms requirements |
| **QPS** | Optimized for infrequent access, not high-throughput |
| **Indexing** | Managed/opaque - no parameter tuning |

### Integration Patterns

#### With Amazon Bedrock Knowledge Bases

```
S3 (Documents) → Bedrock KB → S3 Vectors
       ↓                           ↓
   Raw files              Auto-generated embeddings
```

#### Tiered Architecture with OpenSearch

```
┌─────────────────────────────────────────────────────────┐
│                    Query Router                         │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  OpenSearch   │ │  OpenSearch   │ │  S3 Vectors   │
│   (Hot)       │ │   (Warm)      │ │   (Cold)      │
│   <10ms       │ │   <50ms       │ │   <500ms      │
│   High QPS    │ │   Med QPS     │ │   Low QPS     │
│   $$$$        │ │   $$$         │ │   $           │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## Precision and Recall Experiments

Understanding the accuracy-speed trade-off is critical for production systems.

### Definitions

```
                    Relevant Items
                   ┌─────────────────┐
                   │                 │
Retrieved Items    │    True         │
┌──────────────────┤   Positives     │
│                  │     (TP)        │
│   False          │                 │
│  Positives       └─────────────────┘
│    (FP)
└──────────────────

Precision = TP / (TP + FP)  → "Of what I retrieved, how many are relevant?"
Recall    = TP / (TP + FN)  → "Of all relevant items, how many did I retrieve?"
```

### Recall@K

The standard metric for vector search quality:

```
Recall@K = |Retrieved_K ∩ True_Neighbors_K| / K

Example:
- True top-10 neighbors: {A, B, C, D, E, F, G, H, I, J}
- ANN returned top-10:   {A, B, C, D, E, F, G, X, Y, Z}
- Recall@10 = 7/10 = 0.70
```

### Experiment Design

#### Variables to Test

1. **Dimensions**: 384, 768, 1024, 1536
2. **Index parameters**: M, ef_construction, ef_search, lists, probes
3. **Dataset sizes**: 10K, 100K, 1M vectors
4. **Query patterns**: Random, clustered, filtered

#### Metrics to Capture

- Recall@1, Recall@10, Recall@100
- Query latency (p50, p95, p99)
- Index build time
- Memory/storage usage
- QPS (queries per second)

### Expected Results

#### Recall vs Dimensions

```
Dimension | Recall@10 | Latency (ms) | Memory/Vector
----------|-----------|--------------|---------------
384       | 0.92      | 2.1          | 1.5 KB
768       | 0.95      | 3.8          | 3.1 KB
1024      | 0.97      | 5.2          | 4.1 KB
1536      | 0.98      | 8.1          | 6.1 KB
```

#### Recall vs ef_search (HNSW)

```
ef_search | Recall@10 | Latency (ms)
----------|-----------|-------------
32        | 0.85      | 1.2
64        | 0.92      | 2.1
128       | 0.96      | 4.3
256       | 0.98      | 8.7
512       | 0.99      | 17.2
```

#### Recall vs probes (IVFFlat)

```
probes | Recall@10 | Latency (ms)
-------|-----------|-------------
1      | 0.45      | 0.8
4      | 0.72      | 1.9
16     | 0.89      | 6.2
64     | 0.96      | 23.1
256    | 0.99      | 89.4
```

---

## When to Use What

### Decision Matrix

| Requirement | Recommendation |
|-------------|----------------|
| < 1M vectors, familiar with PostgreSQL | **pgvector** |
| Need hybrid search (text + vector) | **OpenSearch** |
| High QPS (> 1000), low latency (< 10ms) | **OpenSearch** |
| Cost-sensitive, infrequent queries | **S3 Vectors** |
| RAG with Bedrock | **S3 Vectors** |
| Billions of vectors | **OpenSearch** or **S3 Vectors** |
| Need SQL joins with vectors | **pgvector** |
| Already on AWS, want managed | **OpenSearch Service** or **S3 Vectors** |

### Architecture Patterns

#### Pattern 1: Simple RAG

```
User Query → Embedding Model → S3 Vectors → Top-K docs → LLM → Response
```

#### Pattern 2: High-Performance Search

```
User Query → Embedding Model → OpenSearch (HNSW) → Reranker → Results
```

#### Pattern 3: OLTP + Vectors

```
Application → PostgreSQL + pgvector → Combined SQL + Vector queries
```

#### Pattern 4: Cost-Optimized Tiering

```
                    ┌─────────────────┐
                    │  Query Router   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐        ┌───────────┐
   │OpenSearch│         │ pgvector │        │S3 Vectors │
   │  (Hot)   │         │ (Warm)   │        │  (Cold)   │
   │ Frequent │         │ Regular  │        │ Archive   │
   │ queries  │         │ queries  │        │ queries   │
   └─────────┘         └──────────┘        └───────────┘
```

---

## References

- [OpenSearch k-NN Documentation](https://docs.opensearch.org/latest/vector-search/)
- [HNSW Hyperparameter Guide](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [pgvector Indexing Deep Dive](https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/)
- [Amazon S3 Vectors Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [S3 Vectors Getting Started](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html)
- [S3 Vectors Boto3 Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3vectors.html)
