from __future__ import annotations

import math
from typing import Iterable, List, Set


def recall_at_k(
    recommendations: Iterable[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Recall@K:

    Number of relevant items retrieved in top-K
    divided by total relevant items.
    """

    if not relevant_items:
        return 0.0

    recommended = list(recommendations)[:k]

    hits = len(set(recommended) & relevant_items)

    return hits / len(relevant_items)


def hit_rate_at_k(
    recommendations: Iterable[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    HitRate@K:

    1 if at least one relevant item appears
    in the top-K recommendations, otherwise 0.
    """

    if not relevant_items:
        return 0.0

    recommended = list(recommendations)[:k]

    return float(
        bool(set(recommended) & relevant_items)
    )


def ndcg_at_k(
    recommendations: Iterable[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    NDCG@K for implicit-feedback recommendation.

    Relevant items receive gain 1.
    """

    if not relevant_items:
        return 0.0

    recommended = list(recommendations)[:k]

    dcg = 0.0

    for rank, item_id in enumerate(recommended, start=1):

        if item_id in relevant_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_user(
    recommendations: Iterable[int],
    relevant_items: Set[int],
    k: int = 10,
) -> dict:

    return {
        f"recall@{k}": recall_at_k(
            recommendations,
            relevant_items,
            k,
        ),
        f"hit_rate@{k}": hit_rate_at_k(
            recommendations,
            relevant_items,
            k,
        ),
        f"ndcg@{k}": ndcg_at_k(
            recommendations,
            relevant_items,
            k,
        ),
    }