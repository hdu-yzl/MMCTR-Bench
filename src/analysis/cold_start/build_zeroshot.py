#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_zeroshot.py - 构建 Zeroshot 冷启动测试数据集

从原始 Antm2c TFRecord 数据中，收集训练集出现的所有用户和物品 ID，
筛选 val/test 中新用户或新物品（不在训练集中出现）的样本。

数据格式（与 Dtfloader_Antm2c / 6.to_tf.py 一致）:
  - text_features:  float[4608]  (6×768 BERT embeddings 拼接)
  - image_features: float[512]   (CLIP image embedding)
  - id_feature:     int64[2]     ([user_id, item_id])
  - label:          float[1]     (点击标签)
  - domain:         float[1]     (场景类别)
  - user_seq:       int64[5]     (用户最近5次点击物品序列)

用法:
  <SERVER_ENV>/bin/python build_zeroshot.py \
    --src_dir <ANTM2C_DATA_DIR> \
    --dst_dir <OUTPUT_DIR>/zeroshot
"""

import os
import glob
import shutil
import argparse

import tensorflow as tf
from tqdm import tqdm

# TFRecord 维度常量
ID_FIELDS = 2

# 仅解析 id_feature，用于高效过滤
id_description = {
    "id_feature": tf.io.FixedLenFeature([ID_FIELDS], tf.int64),
}


def collect_train_ids(src_dir):
    """收集训练集中出现的所有用户和物品 ID"""
    train_users = set()
    train_items = set()

    train_files = sorted(glob.glob(os.path.join(src_dir, 'train*.tfrecord')))
    if not train_files:
        raise FileNotFoundError(f"未找到训练集 TFRecord: {src_dir}/train*.tfrecord")

    print(f"读取训练集文件: {train_files}")
    dataset = tf.data.TFRecordDataset(train_files)

    total = 0
    for raw_record in tqdm(dataset, desc="收集训练集 ID"):
        example = tf.io.parse_single_example(raw_record, id_description)
        train_users.add(example['id_feature'][0].numpy())
        train_items.add(example['id_feature'][1].numpy())
        total += 1

    print(f"训练集总交互数: {total}")
    print(f"训练集用户数: {len(train_users)}, 物品数: {len(train_items)}")
    return train_users, train_items


def filter_and_write(src_dir, dst_dir, data_type, train_users, train_items):
    """筛选新用户或新物品的样本并写入新 TFRecord"""
    files = sorted(glob.glob(os.path.join(src_dir, f'{data_type}*.tfrecord')))
    if not files:
        print(f"警告: 未找到 {data_type} 文件，跳过")
        return

    dataset = tf.data.TFRecordDataset(files)
    out_path = os.path.join(dst_dir, f'{data_type}.tfrecord')
    writer = tf.io.TFRecordWriter(out_path)

    total, kept = 0, 0
    new_user_count, new_item_count, both_new = 0, 0, 0

    for raw_record in tqdm(dataset, desc=f"筛选 {data_type}"):
        total += 1
        example = tf.io.parse_single_example(raw_record, id_description)
        uid = example['id_feature'][0].numpy()
        iid = example['id_feature'][1].numpy()

        is_new_user = uid not in train_users
        is_new_item = iid not in train_items

        if is_new_user or is_new_item:
            writer.write(raw_record.numpy())
            kept += 1
            if is_new_user:
                new_user_count += 1
            if is_new_item:
                new_item_count += 1
            if is_new_user and is_new_item:
                both_new += 1

    writer.close()

    if total > 0:
        print(f"  {data_type}: 保留 {kept}/{total} 条记录 ({kept / total * 100:.2f}%)")
        print(f"    新用户交互: {new_user_count}, 新物品交互: {new_item_count}, 两者均新: {both_new}")
    else:
        print(f"  {data_type}: 无数据")


def link_or_copy(src, dst):
    """创建软链接，失败则复制文件"""
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
        print(f"  软链接: {os.path.basename(dst)} -> {src}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  复制: {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description='构建 Zeroshot 冷启动数据集 (Antm2c)')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='原始数据目录（包含 train/val/test .tfrecord 和特征 .npy）')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # ========== Step 1: 收集训练集 ID ==========
    print("=" * 60)
    print("[Step 1] 收集训练集用户/物品 ID...")
    train_users, train_items = collect_train_ids(args.src_dir)

    # ========== Step 2: 筛选 val/test ==========
    print("=" * 60)
    print("[Step 2] 筛选 Zeroshot 冷启动样本（新用户或新物品）...")
    for data_type in ['val', 'test']:
        filter_and_write(args.src_dir, args.dst_dir, data_type, train_users, train_items)

    # ========== Step 3: 链接训练数据和共享特征文件 ==========
    print("=" * 60)
    print("[Step 3] 链接训练数据和共享特征文件...")

    # 链接训练集 TFRecord（保持原文件名，loader 用 glob 匹配 train_shuffle*）
    train_files = glob.glob(os.path.join(args.src_dir, 'train*.tfrecord'))
    for f in train_files:
        link_or_copy(f, os.path.join(args.dst_dir, os.path.basename(f)))

    # 链接多模态特征文件（loader 需要 image_feature.npy 和 text_feature.npy）
    for feat_file in ['image_feature.npy', 'text_feature.npy']:
        src = os.path.join(args.src_dir, feat_file)
        if os.path.exists(src):
            link_or_copy(src, os.path.join(args.dst_dir, feat_file))
        else:
            print(f"  警告: {src} 不存在，跳过")

    print("=" * 60)
    print(f"Zeroshot 冷启动数据已保存至: {args.dst_dir}")
    print("使用方法: 修改 config/data.yaml 中 antm2c.data_dir 为该路径即可")


if __name__ == '__main__':
    main()
