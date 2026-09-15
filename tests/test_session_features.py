import numpy as np

from src.features.session_features import (
    build_session_affinity,
    normalize_session_affinity,
)


def test_build_session_affinity_uses_max_similarity():
    candidate_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    recent_item_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scores = build_session_affinity(
        candidate_embeddings,
        recent_item_embeddings,
    )

    expected = np.array(
        [1.0, 1.0, 0.7],
        dtype=np.float32,
    )

    np.testing.assert_allclose(scores, expected)


def test_build_session_affinity_empty_recent_items():
    candidate_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    recent_item_embeddings = np.empty(
        (0, 2),
        dtype=np.float32,
    )

    scores = build_session_affinity(
        candidate_embeddings,
        recent_item_embeddings,
    )

    np.testing.assert_array_equal(
        scores,
        np.zeros(2, dtype=np.float32),
    )


def test_build_session_affinity_empty_candidates():
    candidate_embeddings = np.empty(
        (0, 2),
        dtype=np.float32,
    )

    recent_item_embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    scores = build_session_affinity(
        candidate_embeddings,
        recent_item_embeddings,
    )

    np.testing.assert_array_equal(
        scores,
        np.zeros(0, dtype=np.float32),
    )


def test_normalize_session_affinity():
    scores = np.array(
        [-1.0, 0.0, 1.0],
        dtype=np.float32,
    )

    normalized = normalize_session_affinity(scores)

    expected = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        normalized,
        expected,
    )


def test_normalize_session_affinity_constant_scores():
    scores = np.array(
        [5.0, 5.0, 5.0],
        dtype=np.float32,
    )

    normalized = normalize_session_affinity(scores)

    np.testing.assert_array_equal(
        normalized,
        np.zeros(3, dtype=np.float32),
    )


def test_normalize_session_affinity_empty():
    scores = np.array([], dtype=np.float32)

    normalized = normalize_session_affinity(scores)

    np.testing.assert_array_equal(
        normalized,
        np.array([], dtype=np.float32),
    )
