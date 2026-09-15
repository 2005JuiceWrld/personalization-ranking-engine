from pathlib import Path
import pickle
import re

import pandas as pd
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PART1 = PROJECT_ROOT / "data/raw/retailrocket/item_properties_part1.csv"
PART2 = PROJECT_ROOT / "data/raw/retailrocket/item_properties_part2.csv"
OUTPUT = PROJECT_ROOT / "models/bm25_index.pkl"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_properties(path):
    return pd.read_csv(
        path,
        usecols=["timestamp", "itemid", "property", "value"],
    )


def main():
    print("=" * 60)
    print("BUILDING BM25 ITEM INDEX")
    print("=" * 60)

    df1 = load_properties(PART1)
    df2 = load_properties(PART2)

    df = pd.concat([df1, df2], ignore_index=True)

    # Keep the latest observed value for each item/property.
    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    df = df.drop_duplicates(
        subset=["itemid", "property"],
        keep="last",
    )

    documents = []

    for itemid, group in df.groupby("itemid"):
        tokens = [f"itemid_{int(itemid)}"]

        for row in group.itertuples(index=False):
            prop = str(row.property).lower()
            value = str(row.value).lower()

            tokens.extend(tokenize(prop))
            tokens.extend(tokenize(value))

        documents.append(
            {
                "itemid": int(itemid),
                "tokens": tokens,
            }
        )

    documents.sort(key=lambda x: x["itemid"])

    corpus = [x["tokens"] for x in documents]
    item_ids = [x["itemid"] for x in documents]

    print(f"Property rows: {len(df):,}")
    print(f"Unique items: {len(item_ids):,}")

    bm25 = BM25Okapi(corpus)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "item_ids": item_ids,
            },
            f,
        )

    print(f"Saved: {OUTPUT}")

    print("=" * 60)
    print("BM25 INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
