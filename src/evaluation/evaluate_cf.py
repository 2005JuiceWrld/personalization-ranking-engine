from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)
from src.models.cf_recommender import (
    CollaborativeFilteringRecommender,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def main():

    print("=" * 60)
    print("COLLABORATIVE FILTERING EVALUATION")
    print("=" * 60)

    test = pd.read_csv(
        TEST_PATH,
        usecols=["visitorid", "itemid"],
    )

    recommender = CollaborativeFilteringRecommender()

    user_test_items = (
        test
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    recalls = []
    hits = []
    ndcgs = []

    evaluated_users = 0

    for visitorid, actual_items in user_test_items.items():

        recommendations = recommender.recommend(
            visitorid,
            k=10,
        )

        if not recommendations:
            continue

        recalls.append(
            recall_at_k(
                recommendations,
                actual_items,
                k=10,
            )
        )

        hits.append(
            hit_rate_at_k(
                recommendations,
                actual_items,
                k=10,
            )
        )

        ndcgs.append(
            ndcg_at_k(
                recommendations,
                actual_items,
                k=10,
            )
        )

        evaluated_users += 1

    print(f"\nTest users: {len(user_test_items):,}")
    print(f"Users evaluated: {evaluated_users:,}")

    if not evaluated_users:
        print("\nNo users could be evaluated.")
        return

    print("\nCollaborative Filtering Metrics:")

    print(
        f"Recall@10:   "
        f"{sum(recalls) / len(recalls):.6f}"
    )

    print(
        f"HitRate@10:  "
        f"{sum(hits) / len(hits):.6f}"
    )

    print(
        f"NDCG@10:     "
        f"{sum(ndcgs) / len(ndcgs):.6f}"
    )


if __name__ == "__main__":
    main()