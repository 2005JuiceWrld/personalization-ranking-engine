from pathlib import Path

from src.features.item_representation import (
    load_category_hierarchy,
    load_category_metadata,
    build_item_category_paths,
    build_category_vocabulary,
    build_aligned_category_matrix,
    build_category_embedding,
    save_category_embedding,
)

CATEGORY_TREE = "data/raw/retailrocket/category_tree.csv"

METADATA_PATHS = [
    "data/raw/retailrocket/item_properties_part1.csv",
    "data/raw/retailrocket/item_properties_part2.csv",
]

ITEM_MAPPING = "models/two_tower_item_mapping.csv"

CUTOFF_TIMESTAMP = 1441216151563

MATRIX_PATH = "models/item_category_matrix_temporal.npz"
EMBEDDING_PATH = "models/item_category_embedding_temporal.npy"
SVD_PATH = "models/item_category_svd_temporal.joblib"
VOCAB_PATH = "models/item_category_vocabulary_temporal.csv"


def main():
    print("=" * 60)
    print("BUILDING TEMPORAL CATEGORY REPRESENTATION")
    print("=" * 60)

    print("\nLoading category hierarchy...")
    hierarchy = load_category_hierarchy(CATEGORY_TREE)

    print(f"Categories: {len(hierarchy):,}")

    print("\nLoading category metadata...")
    metadata = load_category_metadata(
        METADATA_PATHS,
        cutoff_timestamp=CUTOFF_TIMESTAMP,
    )

    print(f"Category metadata rows: {len(metadata):,}")
    print(
        f"Metadata items: "
        f"{metadata['itemid'].nunique():,}"
    )

    print("\nBuilding category paths...")
    item_category_paths = build_item_category_paths(
        metadata,
        hierarchy,
    )

    print(
        f"Items with category paths: "
        f"{len(item_category_paths):,}"
    )

    print("\nBuilding category vocabulary...")
    vocabulary = build_category_vocabulary(
        item_category_paths
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary):,}"
    )

    print("\nLoading Two-Tower item mapping...")
    item_mapping = __import__("pandas").read_csv(
        ITEM_MAPPING
    )

    print(
        f"Two-Tower items: "
        f"{len(item_mapping):,}"
    )

    print("\nBuilding aligned sparse matrix...")
    category_matrix = build_aligned_category_matrix(
        item_mapping,
        item_category_paths,
        vocabulary,
    )

    print(
        f"Matrix shape: "
        f"{category_matrix.shape}"
    )

    print(
        f"Matrix non-zero entries: "
        f"{category_matrix.nnz:,}"
    )

    print("\nBuilding 32D category embedding...")
    embedding, svd = build_category_embedding(
        category_matrix,
        n_components=32,
        random_state=42,
    )

    print(
        f"Embedding shape: "
        f"{embedding.shape}"
    )

    print(
        f"Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.6f}"
    )

    print("\nSaving artifacts...")

    category_matrix_path = Path(MATRIX_PATH)
    category_matrix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    from scipy.sparse import save_npz

    save_npz(
        category_matrix_path,
        category_matrix,
    )

    save_category_embedding(
        embedding,
        svd,
        EMBEDDING_PATH,
        SVD_PATH,
    )

    import pandas as pd

    vocabulary_df = pd.DataFrame(
        {
            "categoryid": list(vocabulary.keys()),
            "category_index": list(vocabulary.values()),
        }
    )

    vocabulary_df.to_csv(
        VOCAB_PATH,
        index=False,
    )

    print("\nSaved:")
    print(MATRIX_PATH)
    print(EMBEDDING_PATH)
    print(SVD_PATH)
    print(VOCAB_PATH)

    print("\n" + "=" * 60)
    print("TEMPORAL CATEGORY REPRESENTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
