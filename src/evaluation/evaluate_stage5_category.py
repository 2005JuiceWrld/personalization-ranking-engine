from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_user
from src.features.item_representation import (
    build_category_affinity,
    normalize_category_affinity,
)


TEST_PATH = "data/features/ranking_test.csv"
TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

LTR_MODEL_PATH = "models/lightgbm_ranker.txt"
ITEM_MAPPING_PATH = "models/two_tower_item_mapping.csv"
CATEGORY_EMBEDDING_PATH = "models/item_category_embedding_temporal.npy"


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

TRAIN_END = pd.Timestamp(
    "2015-08-18 04:23:01",
    tz="UTC",
)

RECENT_ITEMS = 20

ALPHAS = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def build_recent_history():

    print("\nLoading train + validation history...")

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=[
            "timestamp",
            "visitorid",
            "itemid",
        ],
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=[
            "timestamp",
            "visitorid",
            "itemid",
        ],
    )

    history = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        format="mixed",
        utc=True,
    )

    history["visitorid"] = history["visitorid"].astype(
        np.int64
    )

    history["itemid"] = history["itemid"].astype(
        np.int64
    )

    history = history.sort_values(
        ["visitorid", "timestamp"],
        ascending=[True, False],
    )


    history = history.groupby(
        "visitorid",
        sort=False,
    ).head(RECENT_ITEMS)

    recent = (
        history.groupby("visitorid")["itemid"]
        .apply(list)
        .to_dict()
    )

    print(
        f"Users with pre-test history: "
        f"{len(recent):,}"
    )

    print(
        f"Recent events/user: "
        f"{history.groupby('visitorid').size().mean():.2f}"
    )

    return recent


def build_item_embedding_lookup():

    print("\nLoading temporal category embeddings...")

    embeddings = np.load(
        CATEGORY_EMBEDDING_PATH
    ).astype(np.float32)

    item_mapping = pd.read_csv(
        ITEM_MAPPING_PATH
    )

    item_mapping["itemid"] = item_mapping[
        "itemid"
    ].astype(np.int64)

    item_mapping["item_index"] = item_mapping[
        "item_index"
    ].astype(int)

    item_to_index = dict(
        zip(
            item_mapping["itemid"],
            item_mapping["item_index"],
        )
    )

    print(
        f"Category embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"Two-Tower item mapping: "
        f"{len(item_mapping):,}"
    )

    return embeddings, item_to_index


def recreate_ltr_scores(df):

    timestamp = pd.to_datetime(
        df["user_item_last_timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df["user_item_recency_days"] = (
        (TRAIN_END - timestamp)
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

    model = lgb.Booster(
        model_file=LTR_MODEL_PATH
    )

    X = (
        df[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    df["ltr_score"] = model.predict(X)

    return df


def evaluate_ranking(
    df,
    score_column,
):

    recalls = []
    hits = []
    ndcgs = []

    users = 0

    for _, group in df.groupby(
        "visitorid",
        sort=False,
    ):

        relevant_items = set(
            group.loc[
                group["label"] == 1,
                "itemid",
            ].astype(int)
        )

        if not relevant_items:
            continue

        ranked = (
            group
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(10)
        )

        items = (
            ranked["itemid"]
            .astype(int)
            .tolist()
        )

        result = evaluate_user(
            items,
            relevant_items,
            k=10,
        )

        recalls.append(
            result["recall@10"]
        )

        hits.append(
            result["hit_rate@10"]
        )

        ndcgs.append(
            result["ndcg@10"]
        )

        users += 1

    return {
        "users": users,
        "recall": float(np.mean(recalls)),
        "hit": float(np.mean(hits)),
        "ndcg": float(np.mean(ndcgs)),
    }


def main():

    print("=" * 70)
    print("STAGE 5 - TEMPORAL CATEGORY AFFINITY OFFLINE EXPERIMENT")
    print("=" * 70)

    print("\nLoading fixed ranking test set...")

    df = pd.read_csv(TEST_PATH)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Users: {df['visitorid'].nunique():,}"
    )

    print(
        f"Candidates/user: "
        f"{df.groupby('visitorid').size().iloc[0]}"
    )

    print(
        f"Positive labels: "
        f"{df['label'].sum():,}"
    )

    # ---------------------------------------------------------
    # Existing LTR score — unchanged definition
    # ---------------------------------------------------------

    print("\nRecreating existing LightGBM scores...")

    df = recreate_ltr_scores(df)

    # ---------------------------------------------------------
    # Fixed baselines
    # ---------------------------------------------------------

    print("\nEvaluating HNSW retrieval...")

    retrieval_df = df.copy()
    retrieval_df["retrieval_score_eval"] = (
        -retrieval_df["retrieval_rank"]
    )

    retrieval = evaluate_ranking(
        retrieval_df,
        "retrieval_score_eval",
    )

    print("\nEvaluating existing LTR...")

    ltr = evaluate_ranking(
        df,
        "ltr_score",
    )

    print("\nBASELINES")
    print("-" * 70)

    print(
        f"HNSW  Recall@10={retrieval['recall']:.6f}  "
        f"HitRate@10={retrieval['hit']:.6f}  "
        f"NDCG@10={retrieval['ndcg']:.6f}"
    )

    print(
        f"LTR   Recall@10={ltr['recall']:.6f}  "
        f"HitRate@10={ltr['hit']:.6f}  "
        f"NDCG@10={ltr['ndcg']:.6f}"
    )

    print(
        f"\nLTR score range: "
        f"{df['ltr_score'].min():.6f} "
        f"to "
        f"{df['ltr_score'].max():.6f}"
    )

    # ---------------------------------------------------------
    # Temporal category representation
    # ---------------------------------------------------------

    recent_history = build_recent_history()

    category_embeddings, item_to_index = (
        build_item_embedding_lookup()
    )

    print("\nBuilding category affinity...")

    affinity = np.zeros(
        len(df),
        dtype=np.float32,
    )

    users_with_affinity = 0
    candidates_with_embedding = 0

    for visitorid, positions in df.groupby(
        "visitorid",
        sort=False,
    ).groups.items():

        recent_ids = recent_history.get(
            int(visitorid),
            [],
        )

        recent_indices = [
            item_to_index[itemid]
            for itemid in recent_ids
            if itemid in item_to_index
        ]

        if not recent_indices:
            continue

        recent_vectors = category_embeddings[
            recent_indices
        ]

        candidate_ids = (
            df.loc[positions, "itemid"]
            .astype(int)
            .tolist()
        )

        candidate_indices = [
            item_to_index.get(itemid, -1)
            for itemid in candidate_ids
        ]

        valid_positions = [
            i
            for i, index in enumerate(candidate_indices)
            if index >= 0
        ]

        if not valid_positions:
            continue

        candidate_vectors = category_embeddings[
            [
                candidate_indices[i]
                for i in valid_positions
            ]
        ]

        scores = build_category_affinity(
            candidate_vectors,
            recent_vectors,
        )

        scores = normalize_category_affinity(
            scores
        )

        group_positions = list(positions)

        for local_index, score in zip(
            valid_positions,
            scores,
        ):
            affinity[
                group_positions[local_index]
            ] = score

        users_with_affinity += 1
        candidates_with_embedding += len(
            valid_positions
        )

    df["category_affinity"] = affinity

    print(
        f"Users with category history: "
        f"{users_with_affinity:,}"
    )

    print(
        f"Candidates with category embedding: "
        f"{candidates_with_embedding:,}"
    )

    print(
        f"Category affinity mean: "
        f"{df['category_affinity'].mean():.6f}"
    )

    print(
        f"Category affinity nonzero: "
        f"{(df['category_affinity'] > 0).mean():.2%}"
    )

    # ---------------------------------------------------------
    # Controlled alpha sweep
    # ---------------------------------------------------------

    results = []

    print("\n" + "=" * 70)
    print("CATEGORY AFFINITY SWEEP")
    print("=" * 70)

    for alpha in ALPHAS:

        score_column = (
            f"stage5_score_{str(alpha).replace('.', '_')}"
        )

        df[score_column] = (
            df["ltr_score"]
            + alpha
            * df["category_affinity"]
        )

        result = evaluate_ranking(
            df,
            score_column,
        )

        results.append(
            {
                "alpha": alpha,
                "recall@10": result["recall"],
                "hit_rate@10": result["hit"],
                "ndcg@10": result["ndcg"],
                "delta_recall": (
                    result["recall"]
                    - ltr["recall"]
                ),
                "delta_hit": (
                    result["hit"]
                    - ltr["hit"]
                ),
                "delta_ndcg": (
                    result["ndcg"]
                    - ltr["ndcg"]
                ),
            }
        )

        print(
            f"alpha={alpha:<5} "
            f"Recall={result['recall']:.6f} "
            f"Hit={result['hit']:.6f} "
            f"NDCG={result['ndcg']:.6f} "
            f"| ?R={result['recall'] - ltr['recall']:+.6f} "
            f"?H={result['hit'] - ltr['hit']:+.6f} "
            f"?N={result['ndcg'] - ltr['ndcg']:+.6f}"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("STAGE 5 RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    best = results_df.loc[
        results_df["ndcg@10"].idxmax()
    ]

    print("\nBest alpha by NDCG@10:")
    print(
        f"alpha={best['alpha']}"
    )

    print(
        f"Recall@10={best['recall@10']:.6f}"
    )

    print(
        f"HitRate@10={best['hit_rate@10']:.6f}"
    )

    print(
        f"NDCG@10={best['ndcg@10']:.6f}"
    )

    print("\n" + "=" * 70)
    print("STAGE 5 EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()




