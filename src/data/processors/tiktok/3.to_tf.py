import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

def create_example(id_features, label, user_seq):
    feature = {
        'id_feature': tf.train.Feature(int64_list=tf.train.Int64List(value=id_features)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'user_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=user_seq))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(df, split_name):
    out_file = f"data/processed/tiktok/{split_name}.tfrecord"
    writer = tf.io.TFRecordWriter(out_file)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {out_file}"):
        id_feat = [int(row.user_id), int(row.item_id)]

        user_seq = row.item_seq
        example = create_example(id_feat,
                                 row.label, user_seq)
        writer.write(example)
    writer.close()
    print(f"{out_file} 写完，共 {len(df)} 条")

if __name__ == '__main__':
    files = ['data/raw/tiktok/image_feat.npy', 'data/raw/tiktok/text_feat.npy', 'data/raw/tiktok/audio_feat.npy']
    save_files = ['data/processed/tiktok/image_feat.npy', 'data/processed/tiktok/text_feat.npy', 'data/processed/tiktok/audio_feat.npy']
    for fname, save_file in zip(files, save_files):
        if not os.path.exists(fname):
            print(f'{fname} 不存在，跳过')
            continue

        feat = np.load(fname)  # 加载
        if feat.shape[0] == 6710:
            zeros = np.zeros((1, feat.shape[1]), dtype=feat.dtype)
            feat = np.concatenate([zeros, feat], axis=0)
            np.save(save_file, feat)  # 覆盖保存
            print(f'{save_file} 已插入全 0 行，新形状：{feat.shape}')
        else:
            np.save(save_file, feat)
            print(f'{fname} 行数不是 6710（实际 {feat.shape[0]}），不做处理')

    min_item_id = 9320

    train_df = pd.read_csv('data/processed/tiktok/train.csv')
    val_df = pd.read_csv('data/processed/tiktok/val.csv')
    test_df = pd.read_csv('data/processed/tiktok/test.csv')

    # 原始列：userid,itemid,userseq,label
    rename_map = {
        'userid': 'user_id',
        'itemid': 'item_id',
        'userseq': 'item_seq',
        'label': 'label'
    }
    for df in [train_df, val_df, test_df]:
        df.rename(columns=rename_map, inplace=True)
        df['item_seq'] = df['item_seq'].apply(lambda s: [int(x) for x in s.split(',')])

    write_tfrecord(train_df, "train")
    write_tfrecord(val_df, "val")
    write_tfrecord(test_df, "test")
