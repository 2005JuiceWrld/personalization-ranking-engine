from __future__ import annotations

from scipy.sparse import csr_matrix

from pathlib import Path

import pandas as pd


def load_category_hierarchy(
    path: str | Path,
) -> dict[int, int | None]:
    categories = pd.read_csv(path)

    categories["categoryid"] = categories["categoryid"].astype(int)

    categories["parentid"] = pd.to_numeric(
        categories["parentid"],
        errors="coerce",
    ).astype("Int64")

    hierarchy: dict[int, int | None] = {}

    for row in categories.itertuples(index=False):
        parent = (
            None
            if pd.isna(row.parentid)
            else int(row.parentid)
        )

        hierarchy[int(row.categoryid)] = parent

    return hierarchy


def build_category_path(
    categoryid: int,
    hierarchy: dict[int, int | None],
) -> list[int]:

    path: list[int] = []
    current: int | None = categoryid
    seen: set[int] = set()

    while current is not None:
        if current in seen:
            raise ValueError(
                f"Category hierarchy cycle detected at "
                f"category {current}"
            )

        seen.add(current)
        path.append(current)

        current = hierarchy.get(current)

    path.reverse()

    return path


def load_category_metadata(
    paths: list[str | Path],
    cutoff_timestamp: int | None = None,
) -> pd.DataFrame:
    frames = []

    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=[
                "timestamp",
                "itemid",
                "property",
                "value",
            ],
        )

        frames.append(frame)

    metadata = pd.concat(
        frames,
        ignore_index=True,
    )

    metadata = metadata[
        metadata["property"] == "categoryid"
    ].copy()

    metadata["timestamp"] = pd.to_numeric(
        metadata["timestamp"],
        errors="coerce",
    )

    metadata["itemid"] = metadata["itemid"].astype(int)

    metadata["categoryid"] = pd.to_numeric(
        metadata["value"],
        errors="coerce",
    )

    metadata = metadata.dropna(
        subset=[
            "timestamp",
            "categoryid",
        ]
    )

    metadata["categoryid"] = (
        metadata["categoryid"].astype(int)
    )

    if cutoff_timestamp is not None:
        metadata = metadata[
            metadata["timestamp"] <= cutoff_timestamp
        ]

    metadata = metadata.sort_values(
        ["itemid", "timestamp"]
    )

    metadata = metadata.drop_duplicates(
        subset=["itemid", "categoryid"],
        keep="last",
    )

    return metadata[
        [
            "itemid",
            "timestamp",
            "categoryid",
        ]
    ]


def build_item_category_paths(
    category_metadata: pd.DataFrame,
    hierarchy: dict[int, int | None],
) -> dict[int, list[list[int]]]:

    item_paths: dict[int, list[list[int]]] = {}

    for row in category_metadata.itertuples(index=False):
        itemid = int(row.itemid)
        categoryid = int(row.categoryid)

        path = build_category_path(
            categoryid,
            hierarchy,
        )

        item_paths.setdefault(
            itemid,
            [],
        ).append(path)

    return item_paths

def build_category_vocabulary(
    item_category_paths: dict[int, list[list[int]]],
) -> dict[int, int]:

    category_ids: set[int] = set()

    for paths in item_category_paths.values():
        for path in paths:
            category_ids.update(path)

    return {
        categoryid: index
        for index, categoryid in enumerate(
            sorted(category_ids)
        )
    }

def build_aligned_category_matrix(
    item_mapping: pd.DataFrame,
    item_category_paths: dict[int, list[list[int]]],
    category_vocabulary: dict[int, int],
) -> csr_matrix:

    mapping = item_mapping.sort_values(
        "item_index"
    ).reset_index(drop=True)

    row_indices: list[int] = []
    column_indices: list[int] = []

    for matrix_row, row in enumerate(
        mapping.itertuples(index=False)
    ):
        itemid = int(row.itemid)

        paths = item_category_paths.get(
            itemid,
            [],
        )

        category_indices = {
            category_vocabulary[categoryid]
            for path in paths
            for categoryid in path
            if categoryid in category_vocabulary
        }

        for category_index in category_indices:
            row_indices.append(matrix_row)
            column_indices.append(category_index)

    data = [1] * len(row_indices)

    return csr_matrix(
        (
            data,
            (row_indices, column_indices),
        ),
        shape=(
            len(mapping),
            len(category_vocabulary),
        ),
        dtype="float32",
    )

from sklearn.decomposition import TruncatedSVD
import numpy as np


def build_category_embedding(
    category_matrix: csr_matrix,
    n_components: int = 32,
    random_state: int = 42,
) -> tuple[np.ndarray, TruncatedSVD]:

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )

    embedding = svd.fit_transform(
        category_matrix
    ).astype("float32")

    return embedding, svd

def save_category_embedding(
    embedding: np.ndarray,
    svd: TruncatedSVD,
    embedding_path: str | Path,
    svd_path: str | Path,
) -> None:

    embedding_path = Path(embedding_path)
    svd_path = Path(svd_path)

    embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    svd_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embedding_path,
        embedding,
    )

    import joblib

    joblib.dump(
        svd,
        svd_path,
    )

def normalize_item_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normalized = np.zeros_like(
        embeddings,
        dtype=np.float32,
    )

    nonzero = norms[:, 0] > 0

    normalized[nonzero] = (
        embeddings[nonzero]
        / norms[nonzero]
    )

    return normalized


def build_category_affinity(
    candidate_embeddings: np.ndarray,
    recent_item_embeddings: np.ndarray,
) -> np.ndarray:

    if (
        candidate_embeddings.size == 0
        or recent_item_embeddings.size == 0
    ):
        return np.zeros(
            len(candidate_embeddings),
            dtype=np.float32,
        )

    candidate_normalized = normalize_item_embeddings(
        candidate_embeddings
    )

    recent_normalized = normalize_item_embeddings(
        recent_item_embeddings
    )

    scores = (
        candidate_normalized
        @ recent_normalized.T
    )

    return scores.max(axis=1).astype(
        np.float32
    )


def normalize_category_affinity(
    scores: np.ndarray,
) -> np.ndarray:

    if scores.size == 0:
        return scores.astype(
            np.float32
        )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros(
            len(scores),
            dtype=np.float32,
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    ).astype(np.float32)