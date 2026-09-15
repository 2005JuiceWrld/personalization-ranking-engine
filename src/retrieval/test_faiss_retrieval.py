from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
import torch


MODEL_PATH = "models/two_tower.pt"
INDEX_PATH = "models/two_tower_items.faiss"
ITEM_MAPPING_PATH = "models/two_tower_item_mapping.csv"

USER_ID = 1
K = 10


def main():
    print("=" * 60)
    print("FAISS RETRIEVAL VERIFICATION")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    user_embeddings = (
        checkpoint["user_embedding.weight"]
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    # Load item mapping
    item_mapping = pd.read_csv(
        ITEM_MAPPING_PATH
    )

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Find internal user index
    # Mapping file contains visitorid -> index
    user_row = item_mapping  # placeholder to keep loading explicit

    del user_row

    user_mapping = pd.read_csv(
        "models/two_tower_user_mapping.csv"
    )

    match = user_mapping[
        user_mapping["visitorid"] == USER_ID
    ]

    if match.empty:
        raise ValueError(
            f"User {USER_ID} not found in user mapping."
        )

    user_index = int(match.iloc[0]["user_index"])

    # Get user vector
    user_vector = user_embeddings[
        user_index
    ].reshape(1, -1)

    # Search FAISS
    scores, indices = index.search(
        user_vector,
        K,
    )

    # Convert FAISS internal item indices
    # back to original item IDs.
    recommendations = item_mapping.iloc[
        indices[0]
    ]["itemid"].astype(int).tolist()

    print(f"\nUser: {USER_ID}")
    print(f"FAISS index size: {index.ntotal:,}")

    print("\nFAISS recommendations:")
    print(recommendations)

    print("\nFAISS scores:")
    print(scores[0].tolist())

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
