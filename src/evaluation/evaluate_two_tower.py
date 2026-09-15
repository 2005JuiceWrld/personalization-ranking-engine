from __future__ import annotations

import pandas as pd

from src.models.two_tower_recommender import TwoTowerRecommender
from src.evaluation.metrics import evaluate_user


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def main():
    print("=" * 60)
    print("TWO-TOWER EVALUATION")
    print("=" * 60)

    print("\nLoading test data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    recommender = TwoTowerRecommender()

    # Users that have training history and therefore can be
    # evaluated as personalized recommendations.
    train_users = set(train["visitorid"].unique())

    test_grouped = test.groupby("visitorid")

    recalls = []
    hits = []
    ndcgs = []

    evaluated = 0

    for visitor_id, group in test_grouped:
        if visitor_id not in train_users:
            continue

        recommendations = recommender.recommend(
            visitorid=int(visitor_id),
            k=10,
        )

        if not recommendations:
            continue

        actual_items = set(group["itemid"].astype(int))

        result = evaluate_user(
            recommendations,
            actual_items,
            k=10,
        )

        recalls.append(result["recall@10"])
        hits.append(result["hit_rate@10"])
        ndcgs.append(result["ndcg@10"])

        evaluated += 1

        if evaluated % 1000 == 0:
            print(f"Evaluated users: {evaluated}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTest users: {test['visitorid'].nunique():,}")
    print(f"Users evaluated: {evaluated:,}")

    if evaluated == 0:
        print("\nNo users could be evaluated.")
        return

    print(f"\nRecall@10:  {sum(recalls) / len(recalls):.6f}")
    print(f"HitRate@10: {sum(hits) / len(hits):.6f}")
    print(f"NDCG@10:    {sum(ndcgs) / len(ndcgs):.6f}")


if __name__ == "__main__":
    main()