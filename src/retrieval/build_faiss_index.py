from __future__ import annotations

import os

import faiss
import numpy as np
import torch


MODEL_PATH = "models/two_tower.pt"
INDEX_PATH = "models/two_tower_items.faiss"


def main():
    print("=" * 60)
    print("BUILDING FAISS TWO-TOWER INDEX")
    print("=" * 60)

    print("\nLoading Two-Tower checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    item_embeddings = (
        checkpoint["item_embedding.weight"]
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    print(f"Item embeddings: {item_embeddings.shape}")

    dimension = item_embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Exact inner-product search.
    # This matches the dot-product scoring used by the
    # current Two-Tower recommender.
    index = faiss.IndexFlatIP(dimension)

    print("\nAdding item embeddings to FAISS...")
    index.add(item_embeddings)

    print(f"FAISS index size: {index.ntotal:,}")

    os.makedirs(
        os.path.dirname(INDEX_PATH),
        exist_ok=True,
    )

    faiss.write_index(
        index,
        INDEX_PATH,
    )

    print(f"\nSaved index:")
    print(INDEX_PATH)

    print("\n" + "=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
