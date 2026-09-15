from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate_user


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

K = 10


def main():

    print("=" * 60)
    print("POPULARITY BASELINE EVALUATION")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Build global popularity from TRAIN ONLY.
    popularity = (
        train
        .groupby("itemid")["interaction_weight"]
        .sum()
        .sort_values(ascending=False)
    )

    recommendations = popularity.head(K).index.tolist()

    print(f"\nTop-{K} global recommendations:")
    print(recommendations)

    # Future interactions in test set.
    test_user_items = (
        test
        .groupby("visitorid")["itemid"]
        .apply(set)
    )

    metrics = []

    for user_id, relevant_items in test_user_items.items():

        result = evaluate_user(
            recommendations=recommendations,
            relevant_items=relevant_items,
            k=K,
        )

        metrics.append(result)

    results = pd.DataFrame(metrics)

    print("\nEvaluation users:", len(results))

    print("\nMetrics:")

    print(
        results
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()