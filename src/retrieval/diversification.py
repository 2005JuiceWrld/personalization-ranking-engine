from __future__ import annotations

import numpy as np


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embedding vectors.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)

    return embeddings / norms


def mmr_select(
    item_ids,
    relevance_scores,
    item_embeddings,
    k=10,
    diversity_weight=0.2,
):
    """
    Select k items using Maximal Marginal Relevance.

    MMR balances:
        relevance to the ranking model
        +
        diversity between selected items

    diversity_weight:
        0.0 -> pure relevance ranking
        1.0 -> maximum diversity emphasis
    """

    item_ids = list(item_ids)

    relevance_scores = np.asarray(
        relevance_scores,
        dtype=np.float32,
    )

    item_embeddings = np.asarray(
        item_embeddings,
        dtype=np.float32,
    )

    if len(item_ids) == 0:
        return []

    if len(item_ids) != len(relevance_scores):
        raise ValueError(
            "item_ids and relevance_scores "
            "must have the same length."
        )

    if len(item_ids) != len(item_embeddings):
        raise ValueError(
            "item_ids and item_embeddings "
            "must have the same length."
        )

    k = min(k, len(item_ids))

    embeddings = normalize_embeddings(
        item_embeddings
    )

    # Normalize relevance scores to [0, 1].
    score_min = relevance_scores.min()
    score_max = relevance_scores.max()

    if score_max > score_min:
        relevance = (
            relevance_scores - score_min
        ) / (
            score_max - score_min
        )
    else:
        relevance = np.ones_like(
            relevance_scores
        )

    selected = []

    # Track the maximum similarity of every candidate
    # against the already-selected items.
    max_similarity = np.full(
        len(item_ids),
        -np.inf,
        dtype=np.float32,
    )

    # First item = highest relevance.
    first = int(
        np.argmax(relevance)
    )

    selected.append(first)

    # Calculate similarity against the first item
    # for every candidate at once.
    similarities = (
        embeddings @ embeddings[first]
    )

    max_similarity = similarities.astype(
        np.float32,
        copy=True,
    )

    max_similarity[first] = -np.inf

    while len(selected) < k:

        # Vectorized MMR score calculation.
        mmr_scores = (
            (1.0 - diversity_weight)
            * relevance
            -
            diversity_weight
            * max_similarity
        )

        # Prevent already-selected items from being
        # selected again.
        mmr_scores[selected] = -np.inf

        best_index = int(
            np.argmax(mmr_scores)
        )

        selected.append(best_index)

        # Update maximum similarity for all remaining
        # candidates against the newly selected item.
        similarities = (
            embeddings @ embeddings[best_index]
        )

        max_similarity = np.maximum(
            max_similarity,
            similarities,
        )

        max_similarity[selected] = -np.inf

    return [
        {
            "itemid": int(item_ids[index]),
            "ltr_score": float(
                relevance_scores[index]
            ),
            "mmr_score": float(
                relevance[index]
            ),
        }
        for index in selected
    ]