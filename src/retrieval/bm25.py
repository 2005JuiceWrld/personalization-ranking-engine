from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "models/bm25_index.pkl"


class BM25Retriever:

    def __init__(self, index_path=INDEX_PATH):
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.item_ids = np.asarray(
            data["item_ids"],
            dtype=np.int64,
        )

    def search(self, query, k=100):
        tokens = str(query).lower().split()

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        k = min(k, len(scores))

        indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            {
                "itemid": int(self.item_ids[index]),
                "bm25_score": float(scores[index]),
                "bm25_rank": rank,
            }
            for rank, index in enumerate(
                indices,
                start=1,
            )
        ]


def reciprocal_rank_fusion(
    semantic_items,
    lexical_items,
    k=100,
    rrf_k=60,
):
    scores = {}
    metadata = {}

    for rank, item in enumerate(
        semantic_items,
        start=1,
    ):
        itemid = int(item["itemid"])
        scores[itemid] = (
            scores.get(itemid, 0.0)
            + 1.0 / (rrf_k + rank)
        )
        metadata.setdefault(itemid, {})[
            "semantic_rank"
        ] = rank

    for rank, item in enumerate(
        lexical_items,
        start=1,
    ):
        itemid = int(item["itemid"])
        scores[itemid] = (
            scores.get(itemid, 0.0)
            + 1.0 / (rrf_k + rank)
        )
        metadata.setdefault(itemid, {})[
            "bm25_rank"
        ] = rank

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:k]

    return [
        {
            "itemid": itemid,
            "rrf_score": float(scores[itemid]),
            **metadata[itemid],
        }
        for itemid in ranked
    ]
