import pandas as pd
import numpy as np


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    df = pd.read_parquet(
        "data/raw/microlens/train.parquet",
        columns=["user_id", "item_seq", "item_id", "likes_level", "views_level", "label"]
    )

    df = df.drop(columns=["likes_level", "views_level"])

    # 只保留 item_seq 最后 5 个
    df["item_seq"] = df["item_seq"].apply(
        lambda seq: [0 if i == 0 else int(i) + 1_000_000 for i in seq[-5:]]
    )
    # item_id 统一加 1 000 000
    df["item_id"] = df["item_id"] + 1_000_000

    # 8:1:1 随机划分
    train, valid, test = np.split(
        df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True),
        [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    train.to_parquet("data/processed/microlens/train_processed.parquet", index=False)
    valid.to_parquet("data/processed/microlens/valid_processed.parquet", index=False)
    test.to_parquet("data/processed/microlens/test_processed.parquet", index=False)


if __name__ == "__main__":
    main()
