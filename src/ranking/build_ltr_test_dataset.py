from __future__ import annotations

import os

import faiss
import numpy as np
import pandas as pd
import torch


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

USER_FEATURES_PATH = "data/features/user_features.csv"
ITEM_FEATURES_PATH = "data/features/item_features.csv"
USER_ITEM_FEATURES_PATH = "data/features/user_item_features.csv"

MODEL_PATH = "models/two_tower.pt"
USER_MAPPING_PATH = "models/two_tower_user_mapping.csv"
ITEM_MAPPING_PATH = "models/two_tower_item_mapping.csv"
FAISS_INDEX_PATH = "models/two_tower_items_hnsw.faiss"

OUTPUT_PATH = "data/features/ranking_test.csv"

TOP_K = 100


def load_interactions(path):
    df = pd.read_csv(path)

    df["visitorid"] = df["visitorid"].astype(np.int64)
    df["itemid"] = df["itemid"].astype(np.int64)

    return df


def load_two_tower():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    user_embeddings = (
        checkpoint["user_embedding.weight"]
        .cpu()
        .numpy()
    )

    user_mapping = pd.read_csv(
        USER_MAPPING_PATH
    )

    item_mapping = pd.read_csv(
        ITEM_MAPPING_PATH
    )

    return (
        user_embeddings,
        user_mapping,
        item_mapping,
    )


def main():

    print("=" * 60)
    print("BUILDING TRUE LTR TEST DATASET")
    print("=" * 60)

    print("\nLoading train history...")
    train = load_interactions(TRAIN_PATH)

    print(f"Train rows: {len(train):,}")

    print("\nLoading validation history...")
    validation = load_interactions(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print("\nLoading test interactions...")
    test = load_interactions(TEST_PATH)

    print(f"Test rows: {len(test):,}")

    # ---------------------------------------------------------
    # History available before the test period
    # ---------------------------------------------------------

    history = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    print(
        f"\nPre-test history: "
        f"{len(history):,}"
    )

    seen_items = (
        history.groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    positive_items = (
        test.groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    print(
        f"Test users: "
        f"{len(positive_items):,}"
    )

    # ---------------------------------------------------------
    # Load Two-Tower
    # ---------------------------------------------------------

    print("\nLoading Two-Tower embeddings...")

    (
        user_embeddings,
        user_mapping,
        item_mapping,
    ) = load_two_tower()

    print(
        f"User embeddings: "
        f"{user_embeddings.shape}"
    )

    print(
        f"Item embeddings: "
        f"{len(item_mapping):,}"
    )

    user_to_index = dict(
        zip(
            user_mapping["visitorid"]
            .astype(np.int64),
            user_mapping["user_index"]
            .astype(int),
        )
    )

    # ---------------------------------------------------------
    # Load HNSW
    # ---------------------------------------------------------

    print("\nLoading HNSW index...")

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    print(
        f"Index vectors: "
        f"{index.ntotal:,}"
    )

    # ---------------------------------------------------------
    # Generate candidates
    # ---------------------------------------------------------

    print("\nGenerating test candidates...")

    rows = []

    processed = 0

    for visitorid, positives in positive_items.items():

        if visitorid not in user_to_index:
            continue

        user_index = user_to_index[visitorid]

        user_vector = (
            user_embeddings[user_index]
            .astype(np.float32)
            .reshape(1, -1)
        )

        seen = seen_items.get(
            visitorid,
            set(),
        )

        # Retrieve extra candidates because
        # previously-seen items will be removed.
        search_k = (
            TOP_K
            + len(seen)
            + 20
        )

        scores, indices = index.search(
            user_vector,
            search_k,
        )

        rank = 1

        for item_index, score in zip(
            indices[0],
            scores[0],
        ):

            itemid = int(
                item_mapping.iloc[
                    item_index
                ]["itemid"]
            )

            if itemid in seen:
                continue

            rows.append(
                {
                    "visitorid": int(visitorid),
                    "itemid": itemid,
                    "retrieval_score": float(score),
                    "retrieval_rank": rank,
                    "label": int(
                        itemid in positives
                    ),
                }
            )

            rank += 1

            if rank > TOP_K:
                break

        processed += 1

        if processed % 1000 == 0:
            print(
                f"Users processed: "
                f"{processed:,}"
            )

    candidates = pd.DataFrame(rows)

    print("\nCandidate generation complete.")

    print(
        f"Users processed: "
        f"{processed:,}"
    )

    print(
        f"Candidate rows: "
        f"{len(candidates):,}"
    )

    # ---------------------------------------------------------
    # Load existing feature tables
    # ---------------------------------------------------------

    print("\nLoading ranking features...")

    user_features = pd.read_csv(
        USER_FEATURES_PATH
    )

    item_features = pd.read_csv(
        ITEM_FEATURES_PATH
    )

    user_item_features = pd.read_csv(
        USER_ITEM_FEATURES_PATH
    )

    user_features["visitorid"] = (
        user_features["visitorid"]
        .astype(np.int64)
    )

    item_features["itemid"] = (
        item_features["itemid"]
        .astype(np.int64)
    )

    user_item_features["visitorid"] = (
        user_item_features["visitorid"]
        .astype(np.int64)
    )

    user_item_features["itemid"] = (
        user_item_features["itemid"]
        .astype(np.int64)
    )

    # ---------------------------------------------------------
    # Merge features
    # ---------------------------------------------------------

    candidates = candidates.merge(
        user_features,
        on="visitorid",
        how="left",
    )

    candidates = candidates.merge(
        item_features,
        on="itemid",
        how="left",
    )

    candidates = candidates.merge(
        user_item_features,
        on=[
            "visitorid",
            "itemid",
        ],
        how="left",
    )

    # ---------------------------------------------------------
    # Fill missing values
    # ---------------------------------------------------------

    numeric_columns = candidates.select_dtypes(
        include=[np.number]
    ).columns

    candidates[numeric_columns] = (
        candidates[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\nTest ranking dataset:")

    print(
        f"Rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Users: "
        f"{candidates['visitorid'].nunique():,}"
    )

    print(
        f"Positive labels: "
        f"{candidates['label'].sum():,}"
    )

    print(
        f"Positive rate: "
        f"{candidates['label'].mean():.6f}"
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("TRUE LTR TEST DATASET COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()