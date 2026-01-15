"""
Vector Search Precision/Recall Benchmark

This script measures:
1. Recall@K at different dimensions (384, 768, 1024, 1536)
2. Recall vs index parameters (ef_search, probes, M)
3. Query latency at different recall levels
4. Trade-offs between accuracy and speed

Supports: pgvector, OpenSearch (extendable to S3 Vectors)

Requirements:
    pip install numpy pandas matplotlib sentence-transformers psycopg2-binary tqdm

Usage:
    python benchmark_recall.py --backend pgvector --dataset-size 10000
    python benchmark_recall.py --backend opensearch --dataset-size 10000
"""

import argparse
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    dataset_size: int = 10000
    query_count: int = 100
    k_values: list = None  # [1, 10, 100]
    dimensions: list = None  # [384, 768, 1024, 1536]
    random_seed: int = 42

    def __post_init__(self):
        self.k_values = self.k_values or [1, 10, 50, 100]
        self.dimensions = self.dimensions or [384, 768, 1024]


# =============================================================================
# Synthetic Dataset Generator
# =============================================================================

class SyntheticDataset:
    """
    Generate synthetic vector dataset for benchmarking.

    Creates clustered data to simulate real-world embedding distributions.
    """

    def __init__(self, size: int, dimension: int, n_clusters: int = 50, seed: int = 42):
        self.size = size
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.seed = seed

        np.random.seed(seed)
        self.vectors, self.cluster_ids = self._generate_clustered_vectors()
        self.query_vectors = self._generate_queries()

    def _generate_clustered_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate vectors clustered around random centroids."""
        # Generate cluster centroids
        centroids = np.random.randn(self.n_clusters, self.dimension)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        # Assign vectors to clusters
        cluster_sizes = np.random.multinomial(
            self.size,
            np.ones(self.n_clusters) / self.n_clusters
        )

        vectors = []
        cluster_ids = []

        for cluster_id, (centroid, size) in enumerate(zip(centroids, cluster_sizes)):
            if size == 0:
                continue

            # Generate vectors around centroid with some noise
            noise = np.random.randn(size, self.dimension) * 0.3
            cluster_vectors = centroid + noise

            # Normalize
            cluster_vectors = cluster_vectors / np.linalg.norm(
                cluster_vectors, axis=1, keepdims=True
            )

            vectors.append(cluster_vectors)
            cluster_ids.extend([cluster_id] * size)

        return np.vstack(vectors), np.array(cluster_ids)

    def _generate_queries(self, n_queries: int = 100) -> np.ndarray:
        """Generate query vectors from random points in the dataset."""
        # Sample from existing vectors with small perturbation
        indices = np.random.choice(self.size, n_queries, replace=False)
        queries = self.vectors[indices].copy()

        # Add small noise
        noise = np.random.randn(n_queries, self.dimension) * 0.1
        queries = queries + noise
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        return queries


# =============================================================================
# Ground Truth Calculator
# =============================================================================

def compute_ground_truth(
    dataset: np.ndarray,
    queries: np.ndarray,
    k: int,
    distance_metric: str = "cosine"
) -> np.ndarray:
    """
    Compute exact k-nearest neighbors (brute force).

    Returns array of shape (n_queries, k) with indices of true neighbors.
    """
    n_queries = queries.shape[0]
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)

    for i, query in enumerate(tqdm(queries, desc=f"Computing ground truth (k={k})")):
        if distance_metric == "cosine":
            # Cosine similarity (higher is better)
            similarities = np.dot(dataset, query)
            # Get top-k indices (descending similarity)
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        else:
            # Euclidean distance (lower is better)
            distances = np.linalg.norm(dataset - query, axis=1)
            top_k_indices = np.argsort(distances)[:k]

        ground_truth[i] = top_k_indices

    return ground_truth


# =============================================================================
# Recall Calculator
# =============================================================================

def calculate_recall(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    k: int
) -> float:
    """
    Calculate Recall@K.

    Recall@K = |Retrieved_K ∩ True_K| / K
    """
    recalls = []

    for gt, pred in zip(ground_truth[:, :k], predictions[:, :k]):
        # Count intersection
        intersection = len(set(gt) & set(pred))
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls)


# =============================================================================
# pgvector Benchmark
# =============================================================================

class PgvectorBenchmark:
    """Benchmark runner for pgvector."""

    def __init__(self, config: dict):
        import psycopg2
        self.conn = psycopg2.connect(**config)
        self.dimension = None

    def setup_table(self, dimension: int):
        """Create table with vector column."""
        self.dimension = dimension
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS benchmark_vectors CASCADE;")
            cur.execute(f"""
                CREATE TABLE benchmark_vectors (
                    id SERIAL PRIMARY KEY,
                    embedding vector({dimension})
                );
            """)
            self.conn.commit()

    def insert_vectors(self, vectors: np.ndarray):
        """Bulk insert vectors."""
        from psycopg2.extras import execute_values

        data = [(v.tolist(),) for v in vectors]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO benchmark_vectors (embedding) VALUES %s",
                data,
                template="(%s::vector)"
            )
            self.conn.commit()

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 128):
        """Create HNSW index with specified parameters."""
        with self.conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS benchmark_hnsw_idx;")
            cur.execute(f"""
                CREATE INDEX benchmark_hnsw_idx
                ON benchmark_vectors
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction});
            """)
            self.conn.commit()

    def create_ivfflat_index(self, lists: int = 100):
        """Create IVFFlat index with specified parameters."""
        with self.conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS benchmark_ivfflat_idx;")
            cur.execute(f"""
                CREATE INDEX benchmark_ivfflat_idx
                ON benchmark_vectors
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
            self.conn.commit()

    def search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int = None,
        probes: int = None
    ) -> tuple[list[int], float]:
        """
        Execute similarity search.

        Returns (neighbor_ids, latency_ms)
        """
        with self.conn.cursor() as cur:
            # Set parameters if specified
            if ef_search is not None:
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            if probes is not None:
                cur.execute(f"SET ivfflat.probes = {probes};")

            start = time.perf_counter()

            cur.execute("""
                SELECT id
                FROM benchmark_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query.tolist(), k))

            latency = (time.perf_counter() - start) * 1000

            results = [row[0] - 1 for row in cur.fetchall()]  # -1 for 0-indexed

            return results, latency

    def batch_search(
        self,
        queries: np.ndarray,
        k: int,
        **params
    ) -> tuple[np.ndarray, list[float]]:
        """Execute batch of searches, return predictions and latencies."""
        predictions = []
        latencies = []

        for query in tqdm(queries, desc="Searching"):
            results, latency = self.search(query, k, **params)
            predictions.append(results)
            latencies.append(latency)

        return np.array(predictions), latencies

    def close(self):
        self.conn.close()


# =============================================================================
# OpenSearch Benchmark
# =============================================================================

class OpenSearchBenchmark:
    """Benchmark runner for OpenSearch."""

    def __init__(self, config: dict):
        from opensearchpy import OpenSearch, helpers
        self.client = OpenSearch(
            hosts=[{"host": config["host"], "port": config["port"]}],
            http_auth=config.get("http_auth"),
            use_ssl=config.get("use_ssl", False),
            verify_certs=False
        )
        self.helpers = helpers
        self.index_name = "benchmark_vectors"
        self.dimension = None

    def setup_index(self, dimension: int, m: int = 16, ef_construction: int = 128):
        """Create index with k-NN mapping."""
        self.dimension = dimension

        # Delete if exists
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",  # Use innerproduct for normalized vectors (equivalent to cosine)
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": ef_construction,
                                "m": m
                            }
                        }
                    }
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=index_body)

    def insert_vectors(self, vectors: np.ndarray):
        """Bulk insert vectors."""
        actions = []
        for i, v in enumerate(vectors):
            action = {
                "_index": self.index_name,
                "_id": str(i),
                "_source": {"embedding": v.tolist()}
            }
            actions.append(action)

        self.helpers.bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)

    def set_ef_search(self, ef_search: int):
        """Update ef_search parameter."""
        self.client.indices.put_settings(
            index=self.index_name,
            body={"index": {"knn.algo_param.ef_search": ef_search}}
        )

    def search(self, query: np.ndarray, k: int) -> tuple[list[int], float]:
        """Execute similarity search."""
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query.tolist(),
                        "k": k
                    }
                }
            }
        }

        start = time.perf_counter()
        response = self.client.search(index=self.index_name, body=search_body)
        latency = (time.perf_counter() - start) * 1000

        results = [int(hit["_id"]) for hit in response["hits"]["hits"]]
        return results, latency

    def batch_search(
        self,
        queries: np.ndarray,
        k: int,
        **params
    ) -> tuple[np.ndarray, list[float]]:
        """Execute batch of searches."""
        if "ef_search" in params:
            self.set_ef_search(params["ef_search"])

        predictions = []
        latencies = []

        for query in tqdm(queries, desc="Searching"):
            results, latency = self.search(query, k)
            predictions.append(results)
            latencies.append(latency)

        return np.array(predictions), latencies

    def close(self):
        pass


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(self, backend: str, backend_config: dict, config: BenchmarkConfig):
        self.config = config
        self.backend_name = backend
        self.results = []

        if backend == "pgvector":
            self.backend = PgvectorBenchmark(backend_config)
        elif backend == "opensearch":
            self.backend = OpenSearchBenchmark(backend_config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def run_dimension_experiment(self) -> pd.DataFrame:
        """
        Experiment 1: Recall vs Dimensions

        Tests how embedding dimensionality affects recall.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Recall vs Dimensions")
        print("=" * 60)

        results = []

        for dim in self.config.dimensions:
            print(f"\n--- Dimension: {dim} ---")

            # Generate dataset
            dataset = SyntheticDataset(
                self.config.dataset_size,
                dim,
                seed=self.config.random_seed
            )

            # Setup backend
            if self.backend_name == "pgvector":
                self.backend.setup_table(dim)
                self.backend.insert_vectors(dataset.vectors)
                self.backend.create_hnsw_index(m=16, ef_construction=128)
            else:
                self.backend.setup_index(dim, m=16, ef_construction=128)
                self.backend.insert_vectors(dataset.vectors)

            # Test each k value
            for k in self.config.k_values:
                # Compute ground truth
                ground_truth = compute_ground_truth(
                    dataset.vectors,
                    dataset.query_vectors[:self.config.query_count],
                    k
                )

                # Run ANN search
                params = {"ef_search": 100} if self.backend_name == "pgvector" else {"ef_search": 100}
                predictions, latencies = self.backend.batch_search(
                    dataset.query_vectors[:self.config.query_count],
                    k,
                    **params
                )

                # Calculate metrics
                recall = calculate_recall(ground_truth, predictions, k)
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)

                results.append({
                    "dimension": dim,
                    "k": k,
                    "recall": recall,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency
                })

                print(f"  k={k}: Recall={recall:.4f}, Latency(avg)={avg_latency:.2f}ms")

        return pd.DataFrame(results)

    def run_parameter_experiment(
        self,
        dimension: int = 384,
        index_type: str = "hnsw"
    ) -> pd.DataFrame:
        """
        Experiment 2: Recall vs Index Parameters

        Tests how index parameters affect recall/latency trade-off.
        """
        print("\n" + "=" * 60)
        print(f"EXPERIMENT 2: Recall vs {index_type.upper()} Parameters")
        print("=" * 60)

        # Generate fixed dataset
        dataset = SyntheticDataset(
            self.config.dataset_size,
            dimension,
            seed=self.config.random_seed
        )

        # Compute ground truth once
        k = 10
        ground_truth = compute_ground_truth(
            dataset.vectors,
            dataset.query_vectors[:self.config.query_count],
            k
        )

        results = []

        if index_type == "hnsw":
            # Test different M and ef_search combinations
            m_values = [8, 16, 32, 64]
            ef_search_values = [16, 32, 64, 128, 256]

            for m in m_values:
                print(f"\n--- M={m} ---")

                # Rebuild index with new M
                if self.backend_name == "pgvector":
                    self.backend.setup_table(dimension)
                    self.backend.insert_vectors(dataset.vectors)
                    self.backend.create_hnsw_index(m=m, ef_construction=256)
                else:
                    self.backend.setup_index(dimension, m=m, ef_construction=256)
                    self.backend.insert_vectors(dataset.vectors)

                for ef_search in ef_search_values:
                    predictions, latencies = self.backend.batch_search(
                        dataset.query_vectors[:self.config.query_count],
                        k,
                        ef_search=ef_search
                    )

                    recall = calculate_recall(ground_truth, predictions, k)
                    avg_latency = np.mean(latencies)

                    results.append({
                        "m": m,
                        "ef_search": ef_search,
                        "recall": recall,
                        "avg_latency_ms": avg_latency,
                        "qps": 1000 / avg_latency
                    })

                    print(f"  ef_search={ef_search}: Recall={recall:.4f}, "
                          f"Latency={avg_latency:.2f}ms, QPS={1000/avg_latency:.1f}")

        elif index_type == "ivfflat" and self.backend_name == "pgvector":
            # Test different lists and probes combinations
            lists_values = [50, 100, 200, 500]
            probes_percentages = [0.01, 0.02, 0.05, 0.1, 0.2]  # As fraction of lists

            for lists in lists_values:
                print(f"\n--- Lists={lists} ---")

                self.backend.setup_table(dimension)
                self.backend.insert_vectors(dataset.vectors)
                self.backend.create_ivfflat_index(lists=lists)

                for probe_pct in probes_percentages:
                    probes = max(1, int(lists * probe_pct))

                    predictions, latencies = self.backend.batch_search(
                        dataset.query_vectors[:self.config.query_count],
                        k,
                        probes=probes
                    )

                    recall = calculate_recall(ground_truth, predictions, k)
                    avg_latency = np.mean(latencies)

                    results.append({
                        "lists": lists,
                        "probes": probes,
                        "probes_pct": probe_pct,
                        "recall": recall,
                        "avg_latency_ms": avg_latency,
                        "qps": 1000 / avg_latency
                    })

                    print(f"  probes={probes} ({probe_pct*100:.0f}%): "
                          f"Recall={recall:.4f}, Latency={avg_latency:.2f}ms")

        return pd.DataFrame(results)

    def run_scalability_experiment(
        self,
        dimension: int = 384,
        sizes: list = None
    ) -> pd.DataFrame:
        """
        Experiment 3: Recall/Latency vs Dataset Size

        Tests how performance scales with data volume.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Scalability")
        print("=" * 60)

        sizes = sizes or [1000, 5000, 10000, 50000]
        results = []
        k = 10

        for size in sizes:
            print(f"\n--- Dataset Size: {size} ---")

            dataset = SyntheticDataset(size, dimension, seed=self.config.random_seed)

            # Compute ground truth
            ground_truth = compute_ground_truth(
                dataset.vectors,
                dataset.query_vectors[:min(50, self.config.query_count)],
                k
            )

            # Setup and index
            if self.backend_name == "pgvector":
                self.backend.setup_table(dimension)
                self.backend.insert_vectors(dataset.vectors)
                self.backend.create_hnsw_index(m=16, ef_construction=128)
            else:
                self.backend.setup_index(dimension, m=16, ef_construction=128)
                self.backend.insert_vectors(dataset.vectors)

            # Search
            predictions, latencies = self.backend.batch_search(
                dataset.query_vectors[:min(50, self.config.query_count)],
                k,
                ef_search=100
            )

            recall = calculate_recall(ground_truth, predictions, k)
            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99)

            results.append({
                "dataset_size": size,
                "recall": recall,
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "qps": 1000 / avg_latency
            })

            print(f"  Recall={recall:.4f}, Latency(avg)={avg_latency:.2f}ms, "
                  f"QPS={1000/avg_latency:.1f}")

        return pd.DataFrame(results)

    def close(self):
        self.backend.close()


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    df_dimension: pd.DataFrame,
    df_params: pd.DataFrame,
    df_scale: pd.DataFrame,
    output_dir: str = "."
):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Recall vs Dimension
    ax1 = axes[0, 0]
    for k in df_dimension["k"].unique():
        data = df_dimension[df_dimension["k"] == k]
        ax1.plot(data["dimension"], data["recall"], marker="o", label=f"k={k}")
    ax1.set_xlabel("Embedding Dimension")
    ax1.set_ylabel("Recall@K")
    ax1.set_title("Recall vs Embedding Dimension")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency vs Dimension
    ax2 = axes[0, 1]
    for k in df_dimension["k"].unique():
        data = df_dimension[df_dimension["k"] == k]
        ax2.plot(data["dimension"], data["avg_latency_ms"], marker="s", label=f"k={k}")
    ax2.set_xlabel("Embedding Dimension")
    ax2.set_ylabel("Average Latency (ms)")
    ax2.set_title("Latency vs Embedding Dimension")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Recall vs ef_search (HNSW)
    ax3 = axes[1, 0]
    if "ef_search" in df_params.columns:
        for m in df_params["m"].unique():
            data = df_params[df_params["m"] == m]
            ax3.plot(data["ef_search"], data["recall"], marker="o", label=f"M={m}")
        ax3.set_xlabel("ef_search")
        ax3.set_ylabel("Recall@10")
        ax3.set_title("Recall vs ef_search (HNSW)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Recall-Latency Trade-off
    ax4 = axes[1, 1]
    if len(df_scale) > 0:
        ax4.scatter(df_scale["avg_latency_ms"], df_scale["recall"], s=100)
        for _, row in df_scale.iterrows():
            ax4.annotate(
                f"{int(row['dataset_size']):,}",
                (row["avg_latency_ms"], row["recall"]),
                textcoords="offset points",
                xytext=(5, 5)
            )
        ax4.set_xlabel("Average Latency (ms)")
        ax4.set_ylabel("Recall@10")
        ax4.set_title("Recall-Latency Trade-off (by dataset size)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "benchmark_results.png", dpi=150)
    print(f"Saved plots to {output_dir}/benchmark_results.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vector Search Benchmark")
    parser.add_argument(
        "--backend",
        choices=["pgvector", "opensearch"],
        default="pgvector",
        help="Backend to benchmark"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        help="Number of vectors in dataset"
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=100,
        help="Number of queries to run"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files"
    )
    parser.add_argument(
        "--skip-dimension",
        action="store_true",
        help="Skip dimension experiment"
    )
    parser.add_argument(
        "--skip-params",
        action="store_true",
        help="Skip parameter experiment"
    )
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip scalability experiment"
    )

    args = parser.parse_args()

    # Backend configuration
    if args.backend == "pgvector":
        backend_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "vectordb",
            "user": "postgres",
            "password": "postgres"
        }
    else:
        backend_config = {
            "host": "localhost",
            "port": 9200,
            "http_auth": ("admin", "admin"),
            "use_ssl": False
        }

    # Benchmark config
    config = BenchmarkConfig(
        dataset_size=args.dataset_size,
        query_count=args.query_count,
        k_values=[1, 10, 50, 100],
        dimensions=[384, 768, 1024]
    )

    print("=" * 60)
    print(f"Vector Search Benchmark - {args.backend}")
    print("=" * 60)
    print(f"Dataset size: {config.dataset_size}")
    print(f"Query count: {config.query_count}")
    print(f"Dimensions: {config.dimensions}")
    print(f"K values: {config.k_values}")

    # Run benchmarks
    runner = BenchmarkRunner(args.backend, backend_config, config)

    results = {}

    try:
        if not args.skip_dimension:
            results["dimension"] = runner.run_dimension_experiment()

        if not args.skip_params:
            results["params_hnsw"] = runner.run_parameter_experiment(
                dimension=384, index_type="hnsw"
            )
            if args.backend == "pgvector":
                results["params_ivfflat"] = runner.run_parameter_experiment(
                    dimension=384, index_type="ivfflat"
                )

        if not args.skip_scale:
            results["scale"] = runner.run_scalability_experiment(
                dimension=384,
                sizes=[1000, 5000, 10000, 25000]
            )

    finally:
        runner.close()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        csv_path = output_dir / f"benchmark_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    # Generate plots
    if HAS_MATPLOTLIB and results:
        plot_results(
            results.get("dimension", pd.DataFrame()),
            results.get("params_hnsw", pd.DataFrame()),
            results.get("scale", pd.DataFrame()),
            args.output_dir
        )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if "dimension" in results:
        print("\nDimension Experiment (k=10):")
        dim_summary = results["dimension"][results["dimension"]["k"] == 10]
        print(dim_summary[["dimension", "recall", "avg_latency_ms"]].to_string(index=False))

    if "params_hnsw" in results:
        print("\nBest HNSW Configurations (recall > 0.95):")
        good_configs = results["params_hnsw"][results["params_hnsw"]["recall"] > 0.95]
        if len(good_configs) > 0:
            best = good_configs.nsmallest(3, "avg_latency_ms")
            print(best[["m", "ef_search", "recall", "avg_latency_ms"]].to_string(index=False))

    if "scale" in results:
        print("\nScalability Results:")
        print(results["scale"][["dataset_size", "recall", "avg_latency_ms", "qps"]].to_string(index=False))


if __name__ == "__main__":
    main()
