from __future__ import annotations

import os

import faiss
import numpy as np
import torch


MODEL_PATH = "models/two_tower.pt"
INDEX_PATH = "models/two_tower_items_hnsw.faiss"

HNSW_M = 32
EF_CONSTRUCTION = 200


def main():
    print("=" * 60)
    print("BUILDING FAISS HNSW INDEX")
    print("=" * 60)

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

    dimension = item_embeddings.shape[1]

    print(f"\nItem embeddings: {item_embeddings.shape}")
    print(f"Dimension: {dimension}")
    print(f"HNSW M: {HNSW_M}")
    print(f"EF Construction: {EF_CONSTRUCTION}")

    index = faiss.IndexHNSWFlat(
        dimension,
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = EF_CONSTRUCTION

    print("\nAdding item embeddings...")
    index.add(item_embeddings)

    print(f"Index size: {index.ntotal:,}")

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
    print("HNSW INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()