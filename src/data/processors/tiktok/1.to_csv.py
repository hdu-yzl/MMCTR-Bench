import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import csv


random.seed(42)
np.random.seed(42)


def load_seq(file):
    with open(file, 'r', encoding='utf-8') as f:
        return {int(k): v for k, v in json.load(f).items()}

def pad_left(seq, max_len=5, pad_val=-1):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq  # 原本item从0开始，暂补-1


def generate_ctr_data(user_seq_dict, neg_ratio_map=None):
    """
    neg_ratio_map: 负样本倍数选择
    返回 list[ (user_id, item_id, user_seq_str, label) ]
    """
    if neg_ratio_map is None:
        neg_ratio_map = {1: 5, 2: 5}

    samples = []
    for user, seq in tqdm(user_seq_dict.items(), desc="gen ctr"):
        n = len(seq)
        if n == 0:
            continue

        pos_pool = set(seq)  # 该用户已交互集合，用于过滤负样本

        # 预采样负样本时一次性多采一点，避免重复
        def sample_neg(k):
            """采样 k 个不与 pos_pool 冲突的负样本"""
            negs = set()
            while len(negs) < k:
                cand = random.randint(0, 6709)
                if cand not in pos_pool:
                    negs.add(cand)
            return list(negs)

        if n == 1:
            # 5 负样本

            seq_feat = pad_left(seq)
            for neg in sample_neg(neg_ratio_map[1]):
                samples.append((user, neg, ','.join(map(str, seq_feat)), 0))

        elif 2 <= n:
            # 随机留 1 条正样本，其余做序列
            pos_idx = random.randint(0, n - 1)
            item_pos = seq[pos_idx]
            remain = seq[:pos_idx] + seq[pos_idx + 1:]
            seq_feat = pad_left(remain)
            samples.append((user, item_pos, ','.join(map(str, seq_feat)), 1))
            for neg in sample_neg(neg_ratio_map[2]):
                samples.append((user, neg, ','.join(map(str, seq_feat)), 0))

    return samples

if __name__ == '__main__':

    train = load_seq('data/raw/tiktok/train.json')
    val = load_seq('data/raw/tiktok/val.json')
    test = load_seq('data/raw/tiktok/test.json')

    user_seq = defaultdict(list)
    for d in (train, val, test):
        for u, items in d.items():
            user_seq[u].extend(items)

    all_items = list(range(6710))  # 全局物品池
    item_set = set(all_items)

    ctr_data = generate_ctr_data(user_seq)

    with open('data/processed/tiktok/ctr_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userid', 'itemid', 'userseq', 'label'])  # 表头
        for user, item, seq_str, label in ctr_data:
            writer.writerow([user, item, seq_str, label])

    print('done! 样本数:', len(ctr_data))
