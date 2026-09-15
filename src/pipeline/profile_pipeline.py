from __future__ import annotations

import time

from src.pipeline.recommend import RecommendationPipeline


def timed(label, function):

    start = time.perf_counter()

    result = function()

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"{label:<35} "
        f"{elapsed:>10.3f} ms"
    )

    return result


def main():

    print("=" * 60)
    print("RECOMMENDATION PIPELINE PROFILING")
    print("=" * 60)

    pipeline = RecommendationPipeline(
        candidate_k=100
    )

    visitorid = 1

    print("\nSingle-query breakdown:\n")

    candidates = timed(
        "HNSW retrieval",
        lambda: pipeline._retrieve_candidates(
            visitorid
        ),
    )

    features = timed(
        "Feature generation",
        lambda: pipeline._build_features(
            candidates
        ),
    )

    def predict():

        features_copy = features.copy()

        features_copy["ltr_score"] = (
            pipeline.rank_model.predict(
                features_copy[
                    pipeline.FEATURE_COLUMNS
                    if hasattr(
                        pipeline,
                        "FEATURE_COLUMNS",
                    )
                    else [
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
                ]
            )
        )

        return features_copy

    ranked = timed(
        "LightGBM prediction",
        predict,
    )

    def diversify():

        diversification_pool = (
            ranked
            .sort_values(
                "ltr_score",
                ascending=False,
            )
            .head(
                min(50, len(ranked))
            )
            .reset_index(drop=True)
        )

        import numpy as np

        item_embeddings = []

        for itemid in diversification_pool[
            "itemid"
        ]:

            matches = np.where(
                pipeline.index_to_item == itemid
            )[0]

            item_embeddings.append(
                pipeline.item_embeddings[
                    matches[0]
                ]
            )

        item_embeddings = np.asarray(
            item_embeddings,
            dtype=np.float32,
        )

        from src.retrieval.diversification import (
            mmr_select,
        )

        return mmr_select(
            item_ids=diversification_pool[
                "itemid"
            ].tolist(),
            relevance_scores=diversification_pool[
                "ltr_score"
            ].to_numpy(),
            item_embeddings=item_embeddings,
            k=10,
            diversity_weight=0.2,
        )

    timed(
        "MMR diversification",
        diversify,
    )

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()