from __future__ import annotations

import os

import faiss
import numpy as np
import pandas as pd
import torch


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

USER_FEATURES_PATH = "data/features/user_features.csv"
ITEM_FEATURES_PATH = "data/features/item_features.csv"
USER_ITEM_FEATURES_PATH = "data/features/user_item_features.csv"

MODEL_PATH = "models/two_tower.pt"
USER_MAPPING_PATH = "models/two_tower_user_mapping.csv"
ITEM_MAPPING_PATH = "models/two_tower_item_mapping.csv"
FAISS_INDEX_PATH = "models/two_tower_items_hnsw.faiss"

OUTPUT_PATH = "data/features/ranking_train.csv"

TOP_K = 100


def load_two_tower_embeddings():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    user_embeddings = checkpoint["user_embedding.weight"].cpu().numpy()
    item_embeddings = checkpoint["item_embedding.weight"].cpu().numpy()

    user_mapping = pd.read_csv(USER_MAPPING_PATH)
    item_mapping = pd.read_csv(ITEM_MAPPING_PATH)

    return (
        user_embeddings,
        item_embeddings,
        user_mapping,
        item_mapping,
    )


def load_features():
    user_features = pd.read_csv(USER_FEATURES_PATH)
    item_features = pd.read_csv(ITEM_FEATURES_PATH)
    user_item_features = pd.read_csv(USER_ITEM_FEATURES_PATH)

    return user_features, item_features, user_item_features


def load_validation():
    validation = pd.read_csv(VALIDATION_PATH)

    validation["timestamp"] = pd.to_datetime(
        validation["timestamp"],
        format="mixed",
        utc=True,
    )

    validation["visitorid"] = validation["visitorid"].astype(np.int64)
    validation["itemid"] = validation["itemid"].astype(np.int64)

    return validation


def build_seen_items(train):
    return (
        train.groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )


def build_positive_labels(validation):
    return (
        validation.groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )


def build_candidate_rows(
    visitorid,
    candidate_item_indices,
    candidate_scores,
    item_mapping,
    seen_items,
    positive_items,
):
    rows = []

    seen = seen_items.get(visitorid, set())
    positives = positive_items.get(visitorid, set())

    rank = 1

    for item_index, score in zip(
        candidate_item_indices,
        candidate_scores,
    ):
        itemid = int(
            item_mapping.iloc[item_index]["itemid"]
        )

        if itemid in seen:
            continue

        label = int(itemid in positives)

        rows.append(
            {
                "visitorid": visitorid,
                "itemid": itemid,
                "retrieval_score": float(score),
                "retrieval_rank": rank,
                "label": label,
            }
        )

        rank += 1

        if rank > TOP_K:
            break

    return rows


def main():
    print("=" * 60)
    print("BUILDING LTR RANKING DATASET")
    print("=" * 60)

    print("\nLoading training history...")
    train = pd.read_csv(TRAIN_PATH)

    train["visitorid"] = train["visitorid"].astype(np.int64)
    train["itemid"] = train["itemid"].astype(np.int64)

    print(f"Training interactions: {len(train):,}")

    print("\nLoading validation interactions...")
    validation = load_validation()

    print(f"Validation interactions: {len(validation):,}")

    print("\nLoading Two-Tower embeddings...")
    (
        user_embeddings,
        item_embeddings,
        user_mapping,
        item_mapping,
    ) = load_two_tower_embeddings()

    print(
        f"User embeddings: {user_embeddings.shape}"
    )
    print(
        f"Item embeddings: {item_embeddings.shape}"
    )

    print("\nLoading HNSW index...")
    index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"FAISS index vectors: {index.ntotal}")

    print("\nBuilding seen-item history...")
    seen_items = build_seen_items(train)

    print(
        f"Users with history: {len(seen_items):,}"
    )

    print("\nBuilding validation labels...")
    positive_items = build_positive_labels(validation)

    print(
        f"Validation users: {len(positive_items):,}"
    )

    user_to_index = dict(
        zip(
            user_mapping["visitorid"].astype(np.int64),
            user_mapping["user_index"].astype(int),
        )
    )

    print("\nGenerating retrieval candidates...")

    rows = []

    evaluated_users = 0

    for visitorid, positives in positive_items.items():

        if visitorid not in user_to_index:
            continue

        user_index = user_to_index[visitorid]

        user_vector = (
            user_embeddings[user_index]
            .astype(np.float32)
            .reshape(1, -1)
        )

        scores, indices = index.search(
            user_vector,
            TOP_K + len(seen_items.get(visitorid, set())) + 20,
        )

        user_rows = build_candidate_rows(
            visitorid=visitorid,
            candidate_item_indices=indices[0],
            candidate_scores=scores[0],
            item_mapping=item_mapping,
            seen_items=seen_items,
            positive_items=positive_items,
        )

        rows.extend(user_rows)

        evaluated_users += 1

        if evaluated_users % 1000 == 0:
            print(
                f"Users processed: {evaluated_users:,}"
            )

    candidates = pd.DataFrame(rows)

    print("\nCandidate generation complete.")
    print(f"Users: {evaluated_users:,}")
    print(f"Candidate rows: {len(candidates):,}")

    print("\nLoading ranking features...")

    user_features, item_features, user_item_features = (
        load_features()
    )

    user_features["visitorid"] = (
        user_features["visitorid"].astype(np.int64)
    )

    item_features["itemid"] = (
        item_features["itemid"].astype(np.int64)
    )

    user_item_features["visitorid"] = (
        user_item_features["visitorid"].astype(np.int64)
    )

    user_item_features["itemid"] = (
        user_item_features["itemid"].astype(np.int64)
    )

    print("\nMerging user features...")

    candidates = candidates.merge(
        user_features,
        on="visitorid",
        how="left",
    )

    print("Merging item features...")

    candidates = candidates.merge(
        item_features,
        on="itemid",
        how="left",
    )

    print("Merging user-item features...")

    candidates = candidates.merge(
        user_item_features,
        on=["visitorid", "itemid"],
        how="left",
    )

    print("\nFilling missing candidate features...")

    numeric_columns = candidates.select_dtypes(
        include=[np.number]
    ).columns

    candidates[numeric_columns] = (
        candidates[numeric_columns].fillna(0)
    )

    print("\nDataset summary:")
    print(f"Rows: {len(candidates):,}")
    print(f"Columns: {len(candidates.columns)}")
    print(
        f"Positive labels: {candidates['label'].sum():,}"
    )
    print(
        f"Positive rate: "
        f"{candidates['label'].mean():.6f}"
    )

    print("\nFeature columns:")
    for column in candidates.columns:
        print(f"  - {column}")

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
    print("LTR DATASET BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()