import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def create_example(id_features, label, user_seq):
    feature = {
        'id_feature': tf.train.Feature(int64_list=tf.train.Int64List(value=id_features)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'user_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=user_seq))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(df, split_name):
    out_file = f"data/processed/microlens/{split_name}.tfrecord"
    writer = tf.io.TFRecordWriter(out_file)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {out_file}"):
        id_feat = [int(row.user_id), int(row.item_id)]

        user_seq = row.item_seq

        example = create_example(id_feat,
                                 row.label, user_seq)
        writer.write(example)
    writer.close()
    print(f"{out_file} 写完，共 {len(df)} 条")


if __name__ == "__main__":
    train_df = pd.read_parquet("data/processed/microlens/train_processed.parquet", columns=[
        "user_id", "item_seq", "item_id", "label"
    ])
    val_df = pd.read_parquet("data/processed/microlens/valid_processed.parquet", columns=[
        "user_id", "item_seq", "item_id", "label"
    ])
    test_df = pd.read_parquet("data/processed/microlens/test_processed.parquet", columns=[
        "user_id", "item_seq", "item_id", "label"
    ])

    image_feat = np.load('data/processed/microlens/img_emb.npy')
    text_feat = np.load('data/processed/microlens/text_emb.npy')

    min_item_id = 1000001
    min_user_id = 1
    max_user_id = 1000000

    write_tfrecord(train_df, "train")
    write_tfrecord(val_df, "val")
    write_tfrecord(test_df, "test")
