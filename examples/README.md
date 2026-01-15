# Vector Search Examples

Working code examples for OpenSearch, pgvector, and Amazon S3 Vectors.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python opensearch_example.py
python pgvector_example.py
python s3_vectors_example.py

# Run benchmarks
python benchmark_recall.py --backend pgvector --dataset-size 10000
```

## Prerequisites

### pgvector
```bash
# PostgreSQL with pgvector extension
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### OpenSearch
```bash
# Local OpenSearch with k-NN plugin
docker run -d --name opensearch \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin123!" \
  -p 9200:9200 \
  opensearchproject/opensearch:latest
```

### S3 Vectors
- Configure AWS credentials: `aws configure`
- Ensure S3 Vectors preview access in your region
- Enable Amazon Bedrock and Titan model access

## Files

| File | Description |
|------|-------------|
| `opensearch_example.py` | HNSW indexing, hybrid search, filtering |
| `pgvector_example.py` | HNSW/IVFFlat indexes, SQL integration |
| `s3_vectors_example.py` | Vector buckets, metadata filtering |
| `benchmark_recall.py` | Precision/recall experiments |

## Benchmark Usage

```bash
# Full benchmark suite
python benchmark_recall.py --backend pgvector --dataset-size 10000

# Skip specific experiments
python benchmark_recall.py --backend opensearch --skip-scale

# Custom output directory
python benchmark_recall.py --output-dir ./results
```

### Benchmark Experiments

1. **Dimension Experiment**: Recall@K at 384, 768, 1024 dimensions
2. **Parameter Experiment**: Recall vs M, ef_search (HNSW) or lists, probes (IVFFlat)
3. **Scalability Experiment**: Performance at different dataset sizes
