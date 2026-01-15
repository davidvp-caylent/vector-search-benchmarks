"""
S3 Vectors Benchmark

Measures recall and latency for Amazon S3 Vectors service.
Note: S3 Vectors is optimized for cost, not low-latency high-QPS workloads.
"""

import boto3
import json
import time
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    dataset_size: int = 1000  # S3 Vectors is slower, use smaller dataset
    query_count: int = 20
    k_values: list = None
    dimensions: list = None
    region: str = "us-east-1"

    def __post_init__(self):
        self.k_values = self.k_values or [1, 10, 50]
        self.dimensions = self.dimensions or [384, 1024]


class SyntheticDataset:
    """Generate synthetic clustered vectors."""

    def __init__(self, size: int, dimension: int, n_clusters: int = 20, seed: int = 42):
        np.random.seed(seed)
        self.size = size
        self.dimension = dimension

        # Generate clustered data
        centroids = np.random.randn(n_clusters, dimension)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        cluster_sizes = np.random.multinomial(size, np.ones(n_clusters) / n_clusters)

        vectors = []
        for centroid, sz in zip(centroids, cluster_sizes):
            if sz == 0:
                continue
            noise = np.random.randn(sz, dimension) * 0.3
            cluster_vectors = centroid + noise
            cluster_vectors = cluster_vectors / np.linalg.norm(cluster_vectors, axis=1, keepdims=True)
            vectors.append(cluster_vectors)

        self.vectors = np.vstack(vectors).astype(np.float32)

        # Generate queries
        indices = np.random.choice(size, min(50, size), replace=False)
        self.queries = self.vectors[indices] + np.random.randn(len(indices), dimension) * 0.1
        self.queries = (self.queries / np.linalg.norm(self.queries, axis=1, keepdims=True)).astype(np.float32)


def compute_ground_truth(dataset: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-NN using cosine similarity."""
    ground_truth = np.zeros((len(queries), k), dtype=np.int32)
    for i, query in enumerate(queries):
        similarities = np.dot(dataset, query)
        top_k = np.argsort(similarities)[-k:][::-1]
        ground_truth[i] = top_k
    return ground_truth


def calculate_recall(ground_truth: np.ndarray, predictions: list, k: int) -> float:
    """Calculate Recall@K."""
    recalls = []
    for gt, pred in zip(ground_truth[:, :k], predictions):
        pred_set = set(pred[:k]) if len(pred) >= k else set(pred)
        gt_set = set(gt)
        intersection = len(pred_set & gt_set)
        recalls.append(intersection / k)
    return np.mean(recalls)


class S3VectorsBenchmark:
    """Benchmark runner for S3 Vectors."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = boto3.client('s3vectors', region_name=config.region)
        self.bucket_name = f"benchmark-vectors-{''.join(random.choices(string.ascii_lowercase, k=6))}"
        self.index_name = "benchmark"

    def setup(self, dimension: int):
        """Create bucket and index."""
        # Create bucket
        try:
            self.client.create_vector_bucket(vectorBucketName=self.bucket_name)
            print(f"Created bucket: {self.bucket_name}")
        except self.client.exceptions.ConflictException:
            pass

        # Delete existing index if any
        try:
            self.client.delete_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
            time.sleep(2)
        except:
            pass

        # Create index
        self.client.create_index(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            dimension=dimension,
            distanceMetric='cosine',
            dataType='float32'
        )
        print(f"Created index: {self.index_name} (dim={dimension})")
        time.sleep(2)

    def insert_vectors(self, vectors: np.ndarray):
        """Insert vectors in batches."""
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="Inserting"):
            batch = vectors[i:i + batch_size]
            vector_data = [
                {
                    'key': f'vec_{i + j}',
                    'data': {'float32': v.tolist()},
                    'metadata': {'idx': str(i + j)}
                }
                for j, v in enumerate(batch)
            ]
            self.client.put_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                vectors=vector_data
            )
        time.sleep(3)  # Allow indexing

    def search(self, query: np.ndarray, k: int) -> tuple:
        """Execute search and return (results, latency_ms)."""
        start = time.perf_counter()
        response = self.client.query_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            queryVector={'float32': query.tolist()},
            topK=k,
            returnDistance=True
        )
        latency = (time.perf_counter() - start) * 1000

        # Extract indices from keys
        results = []
        for v in response.get('vectors', []):
            key = v['key']
            idx = int(key.split('_')[1])
            results.append(idx)

        return results, latency

    def cleanup(self):
        """Delete resources."""
        try:
            self.client.delete_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
        except:
            pass

        try:
            self.client.delete_vector_bucket(vectorBucketName=self.bucket_name)
            print(f"Deleted bucket: {self.bucket_name}")
        except:
            pass


def run_benchmark(config: BenchmarkConfig):
    """Run S3 Vectors benchmark."""
    print("=" * 60)
    print("S3 Vectors Benchmark")
    print("=" * 60)
    print(f"Dataset size: {config.dataset_size}")
    print(f"Dimensions: {config.dimensions}")
    print(f"K values: {config.k_values}")

    results = []
    benchmark = S3VectorsBenchmark(config)

    try:
        for dim in config.dimensions:
            print(f"\n--- Dimension: {dim} ---")

            # Generate data
            dataset = SyntheticDataset(config.dataset_size, dim)

            # Setup
            benchmark.setup(dim)
            benchmark.insert_vectors(dataset.vectors)

            for k in config.k_values:
                print(f"\n  Testing k={k}...")

                # Compute ground truth
                ground_truth = compute_ground_truth(
                    dataset.vectors,
                    dataset.queries[:config.query_count],
                    k
                )

                # Run searches
                predictions = []
                latencies = []
                for query in tqdm(dataset.queries[:config.query_count], desc=f"  k={k}"):
                    pred, latency = benchmark.search(query, k)
                    predictions.append(pred)
                    latencies.append(latency)

                # Calculate metrics
                recall = calculate_recall(ground_truth, predictions, k)
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)

                results.append({
                    'dimension': dim,
                    'k': k,
                    'recall': recall,
                    'avg_latency_ms': avg_latency,
                    'p95_latency_ms': p95_latency,
                    'cold_latency_ms': latencies[0],
                    'warm_latency_ms': np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
                })

                print(f"  k={k}: Recall={recall:.4f}, Latency(avg)={avg_latency:.0f}ms, "
                      f"Cold={latencies[0]:.0f}ms, Warm={np.mean(latencies[1:]):.0f}ms")

            # Cleanup index for next dimension
            try:
                benchmark.client.delete_index(
                    vectorBucketName=benchmark.bucket_name,
                    indexName=benchmark.index_name
                )
                time.sleep(2)
            except:
                pass

    finally:
        benchmark.cleanup()

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-size', type=int, default=500)
    parser.add_argument('--query-count', type=int, default=10)
    parser.add_argument('--output-dir', default='./benchmark_results_s3vectors')
    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset_size=args.dataset_size,
        query_count=args.query_count,
        dimensions=[384, 1024],
        k_values=[1, 10, 50]
    )

    results = run_benchmark(config)

    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(f"{args.output_dir}/s3vectors_results.csv", index=False)
    print(f"\nSaved results to {args.output_dir}/s3vectors_results.csv")

    # Summary
    print("\n" + "=" * 60)
    print("S3 VECTORS SUMMARY")
    print("=" * 60)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
