# Vector Search Benchmarks: OpenSearch vs pgvector vs S3 Vectors

Comprehensive benchmarks comparing vector search performance across three popular platforms, with working code examples and reproducible results.

## Key Findings

| Platform | Recall@10 | Latency | QPS | Best For |
|----------|-----------|---------|-----|----------|
| **OpenSearch** | 100% | 2.7ms | 370 | Production, hybrid search |
| **pgvector** | 100%* | 1.5ms | 673 | PostgreSQL integration |
| **S3 Vectors** | 94-98% | 134ms | 7 | Cost-optimized, RAG |

*\*With tuned parameters. Defaults achieved only 55-83% recall.*

## Repository Structure

```
├── vector-search-guide.md              # Technical reference documentation
├── blog-post-vector-search.md          # Blog post with analysis
├── BENCHMARK_RESULTS.md                # Detailed results breakdown
│
├── examples/
│   ├── opensearch_example.py           # OpenSearch k-NN working example
│   ├── pgvector_example.py             # pgvector working example
│   ├── s3_vectors_example.py           # S3 Vectors working example
│   ├── benchmark_recall.py             # Main benchmark script
│   ├── benchmark_s3vectors.py          # S3-specific benchmark
│   └── requirements.txt                # Python dependencies
│
├── benchmark_results/                  # pgvector results
│   ├── benchmark_results.png
│   ├── platform_comparison.png
│   └── *.csv
│
├── benchmark_results_opensearch/       # OpenSearch results
│   ├── benchmark_results.png
│   └── *.csv
│
└── benchmark_results_s3vectors/        # S3 Vectors results
    ├── s3vectors_results.png
    └── *.csv
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/davidvp-caylent/vector-search-benchmarks.git
cd vector-search-benchmarks
uv venv .venv && source .venv/bin/activate
uv pip install -r examples/requirements.txt

# Start databases
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name opensearch -e "discovery.type=single-node" -p 9200:9200 opensearchproject/opensearch:latest

# Run benchmarks
python examples/benchmark_recall.py --backend pgvector --dataset-size 10000
python examples/benchmark_recall.py --backend opensearch --dataset-size 10000
python examples/benchmark_s3vectors.py --dataset-size 500  # Requires AWS credentials
```

## Benchmark Methodology

- **Dataset**: Random normalized vectors (simulating embeddings)
- **Dimensions**: 384, 768, 1024
- **K values**: 1, 10, 50, 100
- **Ground truth**: Exact brute-force k-NN
- **Metrics**: Recall@K, latency (p50/p95), QPS

## What We Learned

### pgvector requires tuning
Default HNSW parameters yield 55-83% recall. With `M=32, ef_search=64`, recall hits 100%.

### OpenSearch works out-of-the-box
Achieves 100% recall with `M=16` and minimal configuration.

### S3 Vectors trades speed for cost
50-100x slower than local solutions, but up to 90% cheaper for infrequent queries.

## License

MIT

## Contributing

PRs welcome. Please include benchmark results for any new platforms or configurations.

---

Built by [Caylent](https://caylent.com) — AWS Premier Partner.
