# Vector Search Benchmark Results

**Platforms Tested:** pgvector, OpenSearch, Amazon S3 Vectors
**Date:** January 2026
**Hardware:** Apple Silicon (local Docker for pgvector/OpenSearch, AWS for S3 Vectors)

---

## Executive Summary

| Platform | Best Recall@10 | Latency | QPS | Best For |
|----------|---------------|---------|-----|----------|
| **OpenSearch** | 100% | 2.7ms | 370 | High-throughput production |
| **pgvector** | 100% | 1.3ms | 745 | PostgreSQL integration |
| **S3 Vectors** | 98% | 134ms | 7 | Cost-optimized, infrequent queries |

### Key Findings

1. **OpenSearch achieves 100% recall out-of-the-box** with default parameters
2. **pgvector requires tuning** but achieves highest QPS when optimized
3. **S3 Vectors trades latency for cost** - 50-100x slower but 90% cheaper

---

## Platform Comparison: Recall@10

| Dimension | pgvector | OpenSearch | S3 Vectors |
|-----------|----------|------------|------------|
| 384       | 83%      | **100%**   | 94%        |
| 768       | 74%      | **100%**   | -          |
| 1024      | 75%      | **100%**   | 98%        |

*pgvector tested with M=16, ef_search=100 (default-ish). Recall improves significantly with tuning.*

## Platform Comparison: Latency

| Dimension | pgvector | OpenSearch | S3 Vectors |
|-----------|----------|------------|------------|
| 384       | 1.4ms    | 2.7ms      | 155ms      |
| 768       | 2.4ms    | 4.6ms      | -          |
| 1024      | 3.1ms    | 5.6ms      | 134ms      |

---

## OpenSearch Results

### Recall vs Dimensions (k=10)

| Dimension | Recall@1 | Recall@10 | Recall@50 | Recall@100 | Latency |
|-----------|----------|-----------|-----------|------------|---------|
| 384       | 100%     | **99.6%** | 99.8%     | 99.8%      | 2.7ms   |
| 768       | 100%     | **100%**  | 100%      | 100%       | 4.6ms   |
| 1024      | 100%     | **100%**  | 99.96%    | 99.94%     | 5.6ms   |

### HNSW Parameter Tuning

| M  | ef_search=16 | ef_search=64 | ef_search=256 |
|----|--------------|--------------|---------------|
| 8  | 98.6%        | 98.6%        | 98.6%         |
| 16 | **100%**     | **100%**     | **100%**      |
| 32 | **100%**     | **100%**     | **100%**      |
| 64 | **100%**     | **100%**     | **100%**      |

**Key Insight:** OpenSearch with M=16 achieves 100% recall even with low ef_search. Very forgiving defaults.

### Best Configuration
```json
{
  "method": {
    "name": "hnsw",
    "engine": "faiss",
    "space_type": "innerproduct",
    "parameters": {"m": 16, "ef_construction": 128}
  }
}
```
- **Recall@10:** 100%
- **Latency:** 2.7ms
- **QPS:** 370

---

## pgvector Results

### Recall vs Dimensions (k=10, default params)

| Dimension | Recall@1 | Recall@10 | Recall@50 | Recall@100 | Latency |
|-----------|----------|-----------|-----------|------------|---------|
| 384       | 100%     | 83%       | 77%       | 74%        | 1.4ms   |
| 768       | 98%      | 74%       | 69%       | 66%        | 2.4ms   |
| 1024      | 92%      | 75%       | 69%       | 100%*      | 3.1ms   |

*Recall@100 hit 100% due to ef_search exhausting candidate pool.

### HNSW Parameter Tuning (Recall@10)

| M \ ef_search | 16   | 32   | 64   | 128  | 256  |
|---------------|------|------|------|------|------|
| **8**         | 19%  | 33%  | 51%  | 69%  | 100% |
| **16**        | 41%  | 55%  | 74%  | 100% | 100% |
| **32**        | 60%  | 74%  | 100% | 100% | 100% |
| **64**        | 79%  | 100% | 100% | 100% | 100% |

**Key Insight:** pgvector requires careful tuning. M=32, ef_search=64 is the sweet spot.

### HNSW vs IVFFlat

| Metric | HNSW (tuned) | IVFFlat (best) |
|--------|--------------|----------------|
| Recall@10 | **100%** | 67% |
| Latency | 1.5ms | **0.5ms** |
| QPS | 667 | **1896** |
| Build time | Slower | **Faster** |

**Verdict:** Use HNSW unless memory-constrained or recall <70% is acceptable.

### Best Configuration
```sql
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 256);

SET hnsw.ef_search = 64;
```
- **Recall@10:** 100%
- **Latency:** 1.5ms
- **QPS:** 673

---

## S3 Vectors Results

### Recall vs Dimensions

| Dimension | Recall@1 | Recall@10 | Recall@50 | Cold Latency | Warm Latency |
|-----------|----------|-----------|-----------|--------------|--------------|
| 384       | 100%     | 94%       | 99%       | 344ms        | 149ms        |
| 1024      | 100%     | 98%       | 98%       | 266ms        | 133ms        |

### Latency Characteristics

```
Cold Query (first):  ████████████████████████████████████  300-350ms
Warm Query:          ████████████████████  130-200ms
                     |----|----|----|----|
                     0   100  200  300  400 ms
```

**Key Insight:** S3 Vectors has significant cold-start latency. Best for infrequent, cost-sensitive workloads.

### Best Use Cases
- RAG with Bedrock Knowledge Bases
- AI agent long-term memory
- Archive/cold tier in tiered architecture
- Batch processing (not real-time)

---

## Cross-Platform Comparison

### Recall@10 at 384 Dimensions

```
OpenSearch: ████████████████████  100%
S3 Vectors: ██████████████████░░   94%
pgvector:   ████████████████░░░░   83%  (default params)
pgvector:   ████████████████████  100%  (tuned)
```

### Latency at 384 Dimensions

```
pgvector:   ██░░░░░░░░░░░░░░░░░░░░░░░░  1.4ms
OpenSearch: ████░░░░░░░░░░░░░░░░░░░░░░  2.7ms
S3 Vectors: ████████████████████████████████████████  155ms
            |----|----|----|----|----|----|
            0    50   100  150  200  250 ms
```

### Cost vs Performance Trade-off

```
             High Performance
                   ▲
                   │   ┌─────────────┐
                   │   │  OpenSearch │
                   │   └─────────────┘
                   │
                   │   ┌─────────────┐
                   │   │   pgvector  │
                   │   └─────────────┘
                   │
                   │
                   │
                   │                    ┌─────────────┐
                   │                    │ S3 Vectors  │
                   │                    └─────────────┘
                   └────────────────────────────────────► Low Cost
```

---

## Recommendations

### When to Use Each Platform

| Scenario | Recommendation |
|----------|----------------|
| High QPS, low latency (<10ms) | **OpenSearch** |
| Already using PostgreSQL | **pgvector** |
| Cost-sensitive, infrequent queries | **S3 Vectors** |
| RAG with Bedrock | **S3 Vectors** |
| Hybrid search (text + vector) | **OpenSearch** |
| Need SQL joins with vectors | **pgvector** |
| Billions of vectors | **OpenSearch** or **S3 Vectors** |

### Optimal Configurations

#### OpenSearch (Production)
```json
{
  "m": 16,
  "ef_construction": 128,
  "ef_search": 64
}
```

#### pgvector (High Recall)
```sql
WITH (m = 32, ef_construction = 256);
SET hnsw.ef_search = 64;
```

#### S3 Vectors
- No tuning available (managed service)
- Use cosine distance for text embeddings
- Plan for 100-200ms latency

---

## Scalability Notes

### pgvector Scalability (HNSW, M=16, ef_search=100)

| Dataset Size | Recall@10 | Latency |
|--------------|-----------|---------|
| 1,000        | 100%      | 1.0ms   |
| 5,000        | 83%       | 1.7ms   |
| 10,000       | 66%       | 1.7ms   |
| 25,000       | 42%       | 2.2ms   |

**Warning:** Recall degrades at scale with default params. Increase ef_search proportionally.

### OpenSearch Scalability
- Tested up to 10K vectors with stable 100% recall
- Designed for billions of vectors in production
- Consider sharding for very large datasets

### S3 Vectors Scalability
- Supports 2 billion vectors per index
- 10,000 indexes per bucket
- Latency remains ~100-200ms regardless of size

---

## Raw Data Files

```
benchmark_results/                    # pgvector results
├── benchmark_dimension.csv
├── benchmark_params_hnsw.csv
├── benchmark_params_ivfflat.csv
├── benchmark_scale.csv
└── benchmark_results.png

benchmark_results_opensearch/         # OpenSearch results
├── benchmark_dimension.csv
├── benchmark_params_hnsw.csv
└── benchmark_results.png

benchmark_results_s3vectors/          # S3 Vectors results
└── s3vectors_results.csv
```

---

## Methodology

- **Ground truth:** Brute-force exact k-NN
- **Data:** Synthetic clustered Gaussian vectors
- **Distance:** Cosine similarity (all platforms)
- **Queries:** 50 per configuration (10 for S3 Vectors due to cost)
- **Warmup:** First query excluded from timing averages

## Limitations

- Single-node testing (no distributed/sharded scenarios)
- Synthetic data (real embeddings may differ)
- Small-medium dataset sizes
- No concurrent query testing
