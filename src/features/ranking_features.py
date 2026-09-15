from __future__ import annotations

import pandas as pd


EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 10,
}


def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed",
        utc=True,
    )

    return df


def build_user_features(
    interactions: pd.DataFrame,
) -> pd.DataFrame:

    user_features = (
        interactions
        .groupby("visitorid")
        .agg(
            user_interaction_count=(
                "itemid",
                "count",
            ),
            user_unique_items=(
                "itemid",
                "nunique",
            ),
            user_total_weight=(
                "interaction_weight",
                "sum",
            ),
            user_transactions=(
                "transactionid",
                lambda x: x.notna().sum(),
            ),
        )
        .reset_index()
    )

    return user_features


def build_item_features(
    interactions: pd.DataFrame,
) -> pd.DataFrame:

    item_features = (
        interactions
        .groupby("itemid")
        .agg(
            item_interaction_count=(
                "visitorid",
                "count",
            ),
            item_unique_users=(
                "visitorid",
                "nunique",
            ),
            item_total_weight=(
                "interaction_weight",
                "sum",
            ),
            item_transactions=(
                "transactionid",
                lambda x: x.notna().sum(),
            ),
        )
        .reset_index()
    )

    return item_features


def build_user_item_features(
    interactions: pd.DataFrame,
) -> pd.DataFrame:

    user_item_features = (
        interactions
        .groupby(
            ["visitorid", "itemid"]
        )
        .agg(
            user_item_interactions=(
                "itemid",
                "count",
            ),
            user_item_weight=(
                "interaction_weight",
                "sum",
            ),
            user_item_transactions=(
                "transactionid",
                lambda x: x.notna().sum(),
            ),
            user_item_last_timestamp=(
                "timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    return user_item_features


def build_ranking_features(
    interactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    user_features = build_user_features(
        interactions
    )

    item_features = build_item_features(
        interactions
    )

    user_item_features = build_user_item_features(
        interactions
    )

    return (
        user_features,
        item_features,
        user_item_features,
    )


def main():

    print("=" * 60)
    print("BUILDING RANKING FEATURES")
    print("=" * 60)

    train = load_interactions(
        "data/processed/train.csv"
    )

    print(
        f"\nTraining interactions: "
        f"{len(train):,}"
    )

    user_features, item_features, user_item_features = (
        build_ranking_features(train)
    )

    print(
        f"Users: "
        f"{len(user_features):,}"
    )

    print(
        f"Items: "
        f"{len(item_features):,}"
    )

    print(
        f"User-item pairs: "
        f"{len(user_item_features):,}"
    )

    user_features.to_csv(
        "data/features/user_features.csv",
        index=False,
    )

    item_features.to_csv(
        "data/features/item_features.csv",
        index=False,
    )

    user_item_features.to_csv(
        "data/features/user_item_features.csv",
        index=False,
    )

    print("\nSaved:")
    print("data/features/user_features.csv")
    print("data/features/item_features.csv")
    print("data/features/user_item_features.csv")

    print("\n" + "=" * 60)
    print("RANKING FEATURE BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
