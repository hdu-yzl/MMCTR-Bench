#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stat_fewshot.py - 统计 Fewshot 冷启动数据集的规模

读取 build_fewshot.py 生成的 Fewshot 冷启动数据集（val/test .tfrecord），
统计其中的用户数（去重）、物品数（去重）和交互数（记录条数）。

数据格式（与 Dtfloader_Antm2c / build_fewshot.py 一致）:
  - id_feature: int64[2]  ([user_id, item_id])

用法:
  <SERVER_ENV>/bin/python stat_fewshot.py --data_dir <OUTPUT_DIR>/fewshot
  <SERVER_ENV>/bin/python stat_fewshot.py --data_dir <OUTPUT_DIR>/fewshot --splits val test
"""

import os
import glob
import argparse

import tensorflow as tf
from tqdm import tqdm

# TFRecord 维度常量（与 build_fewshot.py 保持一致）
ID_FIELDS = 2

# 仅解析 id_feature，用于高效统计
id_description = {
    "id_feature": tf.io.FixedLenFeature([ID_FIELDS], tf.int64),
}


def stat_split(data_dir, split):
    """统计单个数据划分（val/test）的用户数、物品数、交互数"""
    files = sorted(glob.glob(os.path.join(data_dir, f'{split}*.tfrecord')))
    if not files:
        print(f"警告: 未找到 {split} 文件（{data_dir}/{split}*.tfrecord），跳过")
        return None

    print(f"读取 {split} 文件: {files}")
    dataset = tf.data.TFRecordDataset(files)

    users = set()
    items = set()
    interactions = 0

    for raw_record in tqdm(dataset, desc=f"统计 {split}"):
        example = tf.io.parse_single_example(raw_record, id_description)
        uid = int(example['id_feature'][0].numpy())
        iid = int(example['id_feature'][1].numpy())
        users.add(uid)
        items.add(iid)
        interactions += 1

    return {
        "split": split,
        "users": users,
        "items": items,
        "interactions": interactions,
    }


def main():
    parser = argparse.ArgumentParser(description='统计 Fewshot 冷启动数据集的用户数/物品数/交互数')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Fewshot 冷启动数据目录')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        help='要统计的数据划分，默认 val test')
    args = parser.parse_args()

    print("=" * 60)
    print(f"[统计 Fewshot 冷启动数据集] 数据目录: {args.data_dir}")
    print("=" * 60)

    results = []
    all_users = set()
    all_items = set()
    all_interactions = 0

    for split in args.splits:
        res = stat_split(args.data_dir, split)
        if res is None:
            continue
        results.append(res)
        all_users |= res["users"]
        all_items |= res["items"]
        all_interactions += res["interactions"]

    if not results:
        print("未统计到任何数据，请检查 --data_dir 与 --splits 是否正确。")
        return

    # 打印各划分统计
    print("")
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    header = f"{'划分':<8} {'用户数':>12} {'物品数':>12} {'交互数':>12}"
    print(header)
    print("-" * 48)
    for res in results:
        print(f"{res['split']:<8} {len(res['users']):>12} {len(res['items']):>12} {res['interactions']:>12}")

    # 汇总（用户/物品去重合并，交互直接累加）
    print("-" * 48)
    print(f"{'合计':<8} {len(all_users):>12} {len(all_items):>12} {all_interactions:>12}")
    print("")
    print("注: 合计的用户数/物品数为各划分去重合并后的结果，交互数为各划分累加。")


if __name__ == '__main__':
    main()
