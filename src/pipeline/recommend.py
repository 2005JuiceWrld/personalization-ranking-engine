from __future__ import annotations
from src.retrieval.diversification import mmr_select
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.models.two_tower import TwoTowerModel, EMBEDDING_DIM
from src.streaming.online_features import OnlineFeatureStore
from src.features.session_features import (
    build_session_affinity,
    normalize_session_affinity,
)
from src.features.item_representation import (
    build_category_affinity,
    normalize_category_affinity,
)
from src.retrieval.bm25 import BM25Retriever, reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

TWO_TOWER_PATH = MODEL_DIR / "two_tower.pt"
USER_MAP_PATH = MODEL_DIR / "two_tower_user_mapping.csv"
ITEM_MAP_PATH = MODEL_DIR / "two_tower_item_mapping.csv"

FAISS_INDEX_PATH = MODEL_DIR / "two_tower_items_hnsw.faiss"
LTR_MODEL_PATH = MODEL_DIR / "lightgbm_ranker.txt"
CATEGORY_EMBEDDING_PATH = MODEL_DIR / "item_category_embedding_temporal.npy"
POPULARITY_PATH = MODEL_DIR / "popularity.csv"

USER_FEATURES_PATH = FEATURE_DIR / "user_features.csv"
ITEM_FEATURES_PATH = FEATURE_DIR / "item_features.csv"
USER_ITEM_FEATURES_PATH = FEATURE_DIR / "user_item_features.csv"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

FEATURE_COLUMNS = [
    "retrieval_score",
    "retrieval_rank",
    "user_interaction_count",
    "user_unique_items",
    "user_total_weight",
    "user_transactions",
    "item_interaction_count",
    "item_unique_users",
    "item_total_weight",
    "item_transactions",
    "user_item_interactions",
    "user_item_weight",
    "user_item_transactions",
    "user_item_recency_days",
]

TRAIN_END = pd.Timestamp(
    "2015-08-18 04:23:01",
    tz="UTC",
)


class RecommendationPipeline:

    def __init__(self, candidate_k: int = 100):

        print("Loading recommendation pipeline...")

        self.candidate_k = candidate_k

        # ---------------------------------------------------------
        # Two-Tower checkpoint
        # ---------------------------------------------------------

        checkpoint = torch.load(
            TWO_TOWER_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.user_embeddings = (
            checkpoint["user_embedding.weight"]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        num_users = self.user_embeddings.shape[0]

        # ---------------------------------------------------------
        # User mapping
        # ---------------------------------------------------------

        user_map = pd.read_csv(USER_MAP_PATH)

        self.user_to_index = dict(
            zip(
                user_map["visitorid"],
                user_map["user_index"],
            )
        )

        # ---------------------------------------------------------
        # Item mapping
        # ---------------------------------------------------------

        item_map = pd.read_csv(ITEM_MAP_PATH)

        self.index_to_item = (
            item_map["itemid"]
            .to_numpy()
        )

        self.item_embeddings = (
            checkpoint["item_embedding.weight"]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        self.item_to_embedding = {
            int(itemid): self.item_embeddings[index]
            for index, itemid in enumerate(
                self.index_to_item
            )
        }
        # ---------------------------------------------------------
        # Temporal category representation
        # ---------------------------------------------------------

        self.category_embeddings = np.load(
            CATEGORY_EMBEDDING_PATH
        ).astype("float32")

        self.item_to_category_embedding = {
            int(itemid): self.category_embeddings[index]
            for index, itemid in enumerate(
                self.index_to_item
            )
        }

        # ---------------------------------------------------------
        # HNSW index
        # ---------------------------------------------------------

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )
	
        # Item embeddings are used for diversification.
        self.item_embeddings = (
            checkpoint["item_embedding.weight"]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        self.item_to_embedding = {
        int(itemid): self.item_embeddings[index]
        for index, itemid in enumerate(
            self.index_to_item
        )
}
        # ---------------------------------------------------------
        # LightGBM ranker
        # ---------------------------------------------------------

        self.rank_model = lgb.Booster(
            model_file=str(LTR_MODEL_PATH)
        )

        # ---------------------------------------------------------
        # Popularity fallback for cold-start users
        # ---------------------------------------------------------

        popularity = pd.read_csv(
            POPULARITY_PATH
        )

        self.popularity_items = (
            popularity["itemid"]
            .astype(int)
            .tolist()
        )

        # ---------------------------------------------------------
        # Ranking features
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Pre-index ranking features for fast inference.
        # Pandas joins are appropriate for offline dataset
        # construction, but dictionaries are much faster for
        # per-request serving.
        # ---------------------------------------------------------

        user_features = pd.read_csv(
            USER_FEATURES_PATH
        )

        item_features = pd.read_csv(
            ITEM_FEATURES_PATH
        )

        user_item_features = pd.read_csv(
            USER_ITEM_FEATURES_PATH
        )

        self.user_features = (
            user_features
            .set_index("visitorid")
            .to_dict("index")
        )

        self.item_features = (
            item_features
            .set_index("itemid")
            .to_dict("index")
        )

        self.user_item_features = (
            user_item_features
            .set_index(
                ["visitorid", "itemid"]
            )
            .to_dict("index")
        )

        # ---------------------------------------------------------
        # Training history for seen-item filtering
        # ---------------------------------------------------------

        train = pd.read_csv(
            TRAIN_PATH,
            usecols=["visitorid", "itemid"],
        )

        self.user_items = (
            train
            .groupby("visitorid")["itemid"]
            .apply(set)
            .to_dict()
        )

        self.online_features = OnlineFeatureStore()
        self.bm25 = BM25Retriever()

        print("Pipeline loaded.")

    def _get_user_vector(self, visitorid: int):

        if visitorid not in self.user_to_index:
            return None

        user_index = self.user_to_index[visitorid]

        return self.user_embeddings[
            user_index
        ].reshape(1, -1)

    def _retrieve_candidates(
        self,
        visitorid: int,
        query: str | None = None,
    ):

        user_vector = self._get_user_vector(
            visitorid
        )

        if user_vector is None:
            return pd.DataFrame()

        semantic_scores, semantic_indices = self.index.search(
            user_vector,
            self.candidate_k,
        )

        semantic_items = []

        seen_items = self.user_items.get(
            visitorid,
            set(),
        )

        for score, index in zip(
            semantic_scores[0],
            semantic_indices[0],
        ):
            if index < 0:
                continue

            itemid = int(
                self.index_to_item[index]
            )

            if itemid in seen_items:
                continue

            semantic_items.append(
                {
                    "itemid": itemid,
                    "retrieval_score": float(score),
                }
            )

        if query and query.strip():
            lexical_items = self.bm25.search(
                query,
                self.candidate_k * 3,
            )

            lexical_items = [
                item
                for item in lexical_items
                if (
                    item["itemid"] not in seen_items
                    and item["itemid"] in self.item_to_embedding
                )
            ]

            fused = reciprocal_rank_fusion(
                semantic_items,
                lexical_items,
                k=self.candidate_k,
            )

            rows = []

            for rank, item in enumerate(
                fused,
                start=1,
            ):
                itemid = int(item["itemid"])

                rows.append(
                    {
                        "visitorid": visitorid,
                        "itemid": itemid,
                        "retrieval_score": float(
                            item["rrf_score"]
                        ),
                        "retrieval_rank": rank,
                    }
                )

            return pd.DataFrame(rows)

        rows = []

        for rank, item in enumerate(
            semantic_items,
            start=1,
        ):
            rows.append(
                {
                    "visitorid": visitorid,
                    "itemid": item["itemid"],
                    "retrieval_score": item["retrieval_score"],
                    "retrieval_rank": rank,
                }
            )

        return pd.DataFrame(rows)

    def _build_features(
        self,
        candidates: pd.DataFrame,
    ):

        if candidates.empty:
            return candidates

        rows = []

        for row in candidates.itertuples(
            index=False
        ):

            visitorid = int(
                row.visitorid
            )

            itemid = int(
                row.itemid
            )

            user = self.user_features.get(
                visitorid,
                {}
            )

            item = self.item_features.get(
                itemid,
                {}
            )

            user_item = (
                self.user_item_features.get(
                    (visitorid, itemid),
                    {}
                )
            )

            timestamp = user_item.get(
                "user_item_last_timestamp"
            )

            if pd.notna(timestamp):

                timestamp = pd.to_datetime(
                    timestamp,
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

                if pd.notna(timestamp):

                    recency_days = (
                        (
                            TRAIN_END
                            - timestamp
                        )
                        .total_seconds()
                        / 86400.0
                    )

                else:
                    recency_days = 9999.0

            else:

                recency_days = 9999.0

            rows.append(
                {
                    "visitorid": visitorid,
                    "itemid": itemid,

                    "retrieval_score":
                        float(row.retrieval_score),

                    "retrieval_rank":
                        int(row.retrieval_rank),

                    "user_interaction_count":
                        user.get(
                            "user_interaction_count",
                            0,
                        ),

                    "user_unique_items":
                        user.get(
                            "user_unique_items",
                            0,
                        ),

                    "user_total_weight":
                        user.get(
                            "user_total_weight",
                            0,
                        ),

                    "user_transactions":
                        user.get(
                            "user_transactions",
                            0,
                        ),

                    "item_interaction_count":
                        item.get(
                            "item_interaction_count",
                            0,
                        ),

                    "item_unique_users":
                        item.get(
                            "item_unique_users",
                            0,
                        ),

                    "item_total_weight":
                        item.get(
                            "item_total_weight",
                            0,
                        ),

                    "item_transactions":
                        item.get(
                            "item_transactions",
                            0,
                        ),

                    "user_item_interactions":
                        user_item.get(
                            "user_item_interactions",
                            0,
                        ),

                    "user_item_weight":
                        user_item.get(
                            "user_item_weight",
                            0,
                        ),

                    "user_item_transactions":
                        user_item.get(
                            "user_item_transactions",
                            0,
                        ),

                    "user_item_recency_days":
                        recency_days,
                }
            )

        return pd.DataFrame(rows)

    def recommend(
        self,
        visitorid: int,
        k: int = 10,
        query: str | None = None,
    ):

        # Cold-start fallback for unknown users.
        if visitorid not in self.user_to_index:
            return [
                {
                    "itemid": itemid,
                    "ltr_score": None,
                    "mmr_score": None,
                }
                for itemid in self.popularity_items[:k]
            ]

        candidates = self._retrieve_candidates(
            visitorid,
            query=query,
        )

        if candidates.empty:
            return []

        features = self._build_features(
            candidates
        )

        features["ltr_score"] = (
            self.rank_model.predict(
                features[FEATURE_COLUMNS]
            )
        )

        # ---------------------------------------------------------
        # Online session affinity
        # ---------------------------------------------------------

        online = self.online_features.get_user_features(
            visitorid
        )

        recent_item_ids = [
            int(itemid)
            for itemid in online["recent_items"]
            if int(itemid) in self.item_to_embedding
        ]

        if recent_item_ids:

            recent_item_embeddings = np.asarray(
                [
                    self.item_to_embedding[itemid]
                    for itemid in recent_item_ids
                ],
                dtype=np.float32,
            )

            candidate_embeddings = np.asarray(
                [
                    self.item_to_embedding[int(itemid)]
                    for itemid in candidates["itemid"]
                ],
                dtype=np.float32,
            )

            session_affinity = build_session_affinity(
                candidate_embeddings,
                recent_item_embeddings,
            )

            session_affinity = normalize_session_affinity(
                session_affinity
            )

            features["session_affinity"] = (
                session_affinity
            )

        else:

            features["session_affinity"] = 0.0

        # ---------------------------------------------------------
        # Temporal category affinity
        # ---------------------------------------------------------

        if recent_item_ids:

            recent_category_embeddings = np.asarray(
                [
                    self.item_to_category_embedding[itemid]
                    for itemid in recent_item_ids
                ],
                dtype=np.float32,
            )

            candidate_category_embeddings = np.asarray(
                [
                    self.item_to_category_embedding[int(itemid)]
                    for itemid in candidates["itemid"]
                ],
                dtype=np.float32,
            )

            category_affinity = build_category_affinity(
                candidate_category_embeddings,
                recent_category_embeddings,
            )

            category_affinity = normalize_category_affinity(
                category_affinity
            )

            features["category_affinity"] = category_affinity

        else:

            features["category_affinity"] = 0.0

        # ---------------------------------------------------------
        # Controlled personalization adjustment
        # ---------------------------------------------------------

        CATEGORY_WEIGHT = 4.0
        SESSION_WEIGHT = 0.15

        features["personalized_score"] = (
            features["ltr_score"]
            + CATEGORY_WEIGHT
            * features["category_affinity"]
            + SESSION_WEIGHT
            * features["session_affinity"]
        )

        # Keep a larger ranked pool before diversification.
        diversification_pool = (
            features
            .sort_values(
                "personalized_score",
                ascending=False,
            )
            .head(
                min(k, len(features))
            )
            .reset_index(
                drop=True
            )
        )

        # Map candidate item IDs to their Two-Tower embeddings.
        item_embeddings = np.asarray(
            [
                self.item_to_embedding[int(itemid)]
                for itemid in diversification_pool["itemid"]
            ],
            dtype=np.float32,
        )

        diversified = mmr_select(
            item_ids=diversification_pool["itemid"].tolist(),
            relevance_scores=diversification_pool["personalized_score"].to_numpy(),
            item_embeddings=item_embeddings,
            k=k,
            diversity_weight=0.2,
        )

        # Restore the original LightGBM score and expose
        # the online personalization signal separately.
        score_lookup = (
            diversification_pool
            .set_index("itemid")
            [["ltr_score", "category_affinity", "session_affinity", "personalized_score"]]
            .to_dict("index")
        )

        for recommendation in diversified:
            itemid = recommendation["itemid"]
            scores = score_lookup[itemid]

            recommendation["ltr_score"] = float(
                scores["ltr_score"]
            )

            recommendation["category_affinity"] = float(
                scores["category_affinity"]
            )

            recommendation["session_affinity"] = float(
                scores["session_affinity"]
            )

            recommendation["personalized_score"] = float(
                scores["personalized_score"]
            )

        return diversified


def main():

    print("=" * 60)
    print("END-TO-END RECOMMENDATION PIPELINE")
    print("=" * 60)

    pipeline = RecommendationPipeline(
        candidate_k=100
    )

    visitorid = 1

    recommendations = pipeline.recommend(
        visitorid=visitorid,
        k=10,
    )

    print(f"\nUser: {visitorid}")

    print("\nFinal recommendations:")

    for rank, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"item={recommendation['itemid']} "
            f"ltr={recommendation['ltr_score']:.6f} "
            f"mmr={recommendation['mmr_score']:.6f}"
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
