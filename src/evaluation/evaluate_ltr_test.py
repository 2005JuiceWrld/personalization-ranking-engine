from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_user


DATA_PATH = "data/features/ranking_test.csv"
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
    print("FINAL HELD-OUT LTR EVALUATION")
    print("=" * 60)

    print("\nLoading test ranking dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(
        f"Users: "
        f"{df['visitorid'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Recreate recency feature exactly as during training
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

    df["user_item_recency_days"] = (
        df["user_item_recency_days"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(9999.0)
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("\nLoading LightGBM model...")

    model = lgb.Booster(
        model_file=MODEL_PATH
    )

    print("Model loaded.")

    # ---------------------------------------------------------
    # Generate predictions
    # ---------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()

    X = (
        X.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    print("\nGenerating LTR scores...")

    df["ltr_score"] = model.predict(X)

    print("LTR scores generated.")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    retrieval_recalls = []
    retrieval_hits = []
    retrieval_ndcgs = []

    ltr_recalls = []
    ltr_hits = []
    ltr_ndcgs = []

    users_evaluated = 0

    print("\nEvaluating users...")

    for visitorid, group in df.groupby(
        "visitorid",
        sort=False,
    ):

        relevant_items = set(
            group.loc[
                group["label"] == 1,
                "itemid",
            ]
            .astype(int)
        )

        if not relevant_items:
            continue

        # -----------------------------------------------------
        # HNSW retrieval ranking
        # -----------------------------------------------------

        retrieval_group = (
            group
            .sort_values("retrieval_rank")
            .head(10)
        )

        retrieval_items = (
            retrieval_group["itemid"]
            .astype(int)
            .tolist()
        )

        retrieval_result = evaluate_user(
            retrieval_items,
            relevant_items,
            k=10,
        )

        retrieval_recalls.append(
            retrieval_result["recall@10"]
        )

        retrieval_hits.append(
            retrieval_result["hit_rate@10"]
        )

        retrieval_ndcgs.append(
            retrieval_result["ndcg@10"]
        )

        # -----------------------------------------------------
        # LightGBM ranking
        # -----------------------------------------------------

        ltr_group = (
            group
            .sort_values(
                "ltr_score",
                ascending=False,
            )
            .head(10)
        )

        ltr_items = (
            ltr_group["itemid"]
            .astype(int)
            .tolist()
        )

        ltr_result = evaluate_user(
            ltr_items,
            relevant_items,
            k=10,
        )

        ltr_recalls.append(
            ltr_result["recall@10"]
        )

        ltr_hits.append(
            ltr_result["hit_rate@10"]
        )

        ltr_ndcgs.append(
            ltr_result["ndcg@10"]
        )

        users_evaluated += 1

        if users_evaluated % 1000 == 0:
            print(
                f"Evaluated users: "
                f"{users_evaluated:,}"
            )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    retrieval_recall = np.mean(
        retrieval_recalls
    )

    retrieval_hit = np.mean(
        retrieval_hits
    )

    retrieval_ndcg = np.mean(
        retrieval_ndcgs
    )

    ltr_recall = np.mean(
        ltr_recalls
    )

    ltr_hit = np.mean(
        ltr_hits
    )

    ltr_ndcg = np.mean(
        ltr_ndcgs
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL HELD-OUT RESULTS")
    print("=" * 60)

    print(
        f"\nUsers evaluated: "
        f"{users_evaluated:,}"
    )

    print("\nHNSW Retrieval Top-10:")
    print(
        f"Recall@10:    "
        f"{retrieval_recall:.6f}"
    )
    print(
        f"HitRate@10:   "
        f"{retrieval_hit:.6f}"
    )
    print(
        f"NDCG@10:      "
        f"{retrieval_ndcg:.6f}"
    )

    print("\nHNSW + LightGBM LTR:")
    print(
        f"Recall@10:    "
        f"{ltr_recall:.6f}"
    )
    print(
        f"HitRate@10:   "
        f"{ltr_hit:.6f}"
    )
    print(
        f"NDCG@10:      "
        f"{ltr_ndcg:.6f}"
    )

    print("\nAbsolute change:")

    print(
        f"Recall@10:    "
        f"{ltr_recall - retrieval_recall:+.6f}"
    )

    print(
        f"HitRate@10:   "
        f"{ltr_hit - retrieval_hit:+.6f}"
    )

    print(
        f"NDCG@10:      "
        f"{ltr_ndcg - retrieval_ndcg:+.6f}"
    )

    print("\n" + "=" * 60)
    print("FINAL LTR EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()