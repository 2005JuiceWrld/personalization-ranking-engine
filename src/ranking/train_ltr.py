from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_PATH = "data/features/ranking_train.csv"
MODEL_PATH = "models/lightgbm_ranker.txt"

FEATURE_COLUMNS = [
    "retrieval_score",
    "retrieval_rank",

    "user_interaction_count",
    "user_unique_items",
    "user_total_weight",
    "user_transactions",

    "item_interaction_count",
    "item_unique_users",
    "item_total_weight",
    "item_transactions",

    "user_item_interactions",
    "user_item_weight",
    "user_item_transactions",

    "user_item_recency_days",
]


def main():
    print("=" * 60)
    print("TRAINING LIGHTGBM LEARNING-TO-RANK MODEL")
    print("=" * 60)

    print("\nLoading ranking dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Convert stored timestamp representation into numeric
    # recency relative to the end of the training period.
    # ---------------------------------------------------------

    train_end = pd.Timestamp(
        "2015-08-18 04:23:01",
        tz="UTC",
    )

    timestamp = pd.to_datetime(
        df["user_item_last_timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["user_item_recency_days"] = (
        (train_end - timestamp)
        .dt.total_seconds()
        / 86400.0
    )

    # Candidates never seen in training receive no history.
    df["user_item_recency_days"] = (
        df["user_item_recency_days"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(9999.0)
    )

    # ---------------------------------------------------------
    # Fill numerical feature values.
    # ---------------------------------------------------------

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # LambdaRank requires groups.
    #
    # Every user represents one ranking query.
    # Each user's 100 candidates form one group.
    # ---------------------------------------------------------

    df = df.sort_values(
        ["visitorid", "retrieval_rank"]
    ).reset_index(drop=True)

    group_sizes = (
        df.groupby("visitorid", sort=False)
        .size()
        .to_numpy()
    )

    print(
        f"\nRanking groups: {len(group_sizes):,}"
    )

    print(
        f"Average candidates/group: "
        f"{group_sizes.mean():.2f}"
    )

    print(
        f"Min candidates/group: "
        f"{group_sizes.min()}"
    )

    print(
        f"Max candidates/group: "
        f"{group_sizes.max()}"
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    y = df["label"].astype(np.int32)

    print(
        f"\nPositive labels: {y.sum():,}"
    )

    print(
        f"Positive rate: {y.mean():.6f}"
    )

    # ---------------------------------------------------------
    # Training matrix
    # ---------------------------------------------------------

    X = df[FEATURE_COLUMNS]

    print("\nTraining features:")

    for column in FEATURE_COLUMNS:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # LightGBM LambdaRank
    # ---------------------------------------------------------

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[10, 50, 100],

        n_estimators=300,
        learning_rate=0.05,

        num_leaves=31,
        max_depth=-1,

        min_child_samples=50,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=42,

        n_jobs=-1,
        verbosity=-1,
    )

    print("\nTraining LambdaRank model...")

    ranker.fit(
        X,
        y,
        group=group_sizes,
    )

    print("\nTraining complete.")

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": ranker.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    ranker.booster_.save_model(
        MODEL_PATH
    )

    print("\nSaved:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("LIGHTGBM LTR TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()