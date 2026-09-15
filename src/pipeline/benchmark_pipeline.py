from __future__ import annotations

import time

from src.pipeline.recommend import RecommendationPipeline


def main():

    print("=" * 60)
    print("END-TO-END RECOMMENDATION PIPELINE BENCHMARK")
    print("=" * 60)

    pipeline = RecommendationPipeline(
        candidate_k=100
    )

    users = list(
        pipeline.user_to_index.keys()
    )[:100]

    # Warm-up
    for visitorid in users[:10]:
        pipeline.recommend(
            visitorid=visitorid,
            k=10,
        )

    start = time.perf_counter()

    for visitorid in users:
        pipeline.recommend(
            visitorid=visitorid,
            k=10,
        )

    elapsed = (
        time.perf_counter() - start
    )

    total_queries = len(users)

    avg_ms = (
        elapsed
        / total_queries
        * 1000
    )

    qps = (
        total_queries / elapsed
    )

    print(
        f"\nQueries: {total_queries:,}"
    )

    print(
        f"Total time: {elapsed:.4f} seconds"
    )

    print(
        f"Average latency: {avg_ms:.3f} ms/query"
    )

    print(
        f"Throughput: {qps:.2f} queries/sec"
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()