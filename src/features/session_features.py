from __future__ import annotations

import numpy as np


def build_session_affinity(
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

    scores = (
        candidate_embeddings
        @ recent_item_embeddings.T
    )

    return scores.max(axis=1).astype(np.float32)


def normalize_session_affinity(
    scores: np.ndarray,
) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)

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