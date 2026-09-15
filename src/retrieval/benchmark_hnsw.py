from __future__ import annotations

import time

import faiss
import numpy as np
import torch


MODEL_PATH = "models/two_tower.pt"
EXACT_INDEX_PATH = "models/two_tower_items.faiss"
HNSW_INDEX_PATH = "models/two_tower_items_hnsw.faiss"

NUM_QUERIES = 1000
K = 10
EF_SEARCH_VALUES = [16, 32, 64, 128]


def measure_latency(index, queries):
    # Warm-up
    index.search(queries[:10], K)

    start = time.perf_counter()

    index.search(queries, K)

    elapsed = time.perf_counter() - start

    return (elapsed / len(queries)) * 1000


def main():
    print("=" * 60)
    print("HNSW LATENCY / RECALL BENCHMARK")
    print("=" * 60)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    queries = (
        checkpoint["user_embedding.weight"]
        [:NUM_QUERIES]
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    exact = faiss.read_index(
        EXACT_INDEX_PATH
    )

    hnsw = faiss.read_index(
        HNSW_INDEX_PATH
    )

    print(f"\nQueries: {len(queries):,}")
    print(f"Items: {exact.ntotal:,}")
    print(f"Dimensions: {exact.d}")

    # Exact nearest neighbors = ground truth.
    _, ground_truth = exact.search(
        queries,
        K,
    )

    print("\nExact baseline")

    exact_latency = measure_latency(
        exact,
        queries,
    )

    print(
        f"Latency: {exact_latency:.3f} ms/query"
    )

    print("\nHNSW results")
    print(
        f"{'efSearch':>10} | "
        f"{'Latency (ms)':>13} | "
        f"{'Recall@10':>10}"
    )

    print("-" * 42)

    for ef_search in EF_SEARCH_VALUES:
        hnsw.hnsw.efSearch = ef_search

        _, predicted = hnsw.search(
            queries,
            K,
        )

        recall_values = []

        for i in range(len(queries)):
            actual = set(ground_truth[i])
            retrieved = set(predicted[i])

            recall = len(
                actual & retrieved
            ) / K

            recall_values.append(recall)

        recall = float(
            np.mean(recall_values)
        )

        latency = measure_latency(
            hnsw,
            queries,
        )

        print(
            f"{ef_search:>10} | "
            f"{latency:>13.3f} | "
            f"{recall:>10.4f}"
        )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()