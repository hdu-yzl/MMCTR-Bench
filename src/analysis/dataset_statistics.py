"""
数据集信息统计脚本
统计: 训练/验证/测试样本数, 总用户数, 总物品数, 多模态维度, 正样本(label=1)数量
支持: TFRecord 模式 和 本地 pkl 模式
"""
import os
import sys
import glob
import yaml
import numpy as np

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def stat_tfrecord(tfrecord_path, description, data_type, id_fields_num):
    """从 TFRecord 文件统计样本数、用户ID集合、物品ID集合、正样本数"""
    import tensorflow as tf

    if data_type == 'train':
        patterns = [
            os.path.join(tfrecord_path, "train_shuffle*.tfrecord"),
            os.path.join(tfrecord_path, "train*.tfrecord"),
        ]
    else:
        patterns = [os.path.join(tfrecord_path, f"{data_type}*.tfrecord")]

    files = []
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            break

    if not files:
        return 0, set(), set(), 0

    total_count = 0
    positive_count = 0
    user_ids = set()
    item_ids = set()

    ds = tf.data.TFRecordDataset(files)
    for raw_record in ds:
        example = tf.io.parse_single_example(raw_record, description)
        ids = example['id_feature'].numpy()
        label = example['label'].numpy()

        user_ids.add(int(ids[0]))
        if id_fields_num >= 2:
            item_ids.add(int(ids[1]))

        if label[0] >= 0.5:
            positive_count += 1
        total_count += 1

    return total_count, user_ids, item_ids, positive_count


def stat_pkl(pkl_path):
    """从本地 pkl 文件统计 (注意: pkl 只保存了前5个batch, 统计结果不完整)"""
    import pickle
    import torch

    if not os.path.exists(pkl_path):
        return 0, set(), set(), 0

    with open(pkl_path, 'rb') as f:
        batches = pickle.load(f)

    total_count = 0
    positive_count = 0
    user_ids = set()
    item_ids = set()

    for batch in batches:
        # batch 格式: (feats, feats_seq, label) 或 (user_feats, feats, feats_seq, label)
        if len(batch) == 3:
            feats, _, label = batch
        elif len(batch) == 4:
            _, feats, _, label = batch
        else:
            continue

        id_feat = feats['id']
        if isinstance(id_feat, torch.Tensor):
            id_feat = id_feat.numpy()

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        for row in id_feat:
            user_ids.add(int(row[0]))
            if len(row) >= 2:
                item_ids.add(int(row[1]))

        positive_count += int((label.flatten() >= 0.5).sum())
        total_count += len(label)

    return total_count, user_ids, item_ids, positive_count


def get_tfrecord_description(dataset_name, data_config):
    """根据数据集名构造 TFRecord 的 description"""
    import tensorflow as tf

    id_fields_num = data_config['id_fields_num']
    seq_len = data_config['seq_len']

    if dataset_name == 'antm2c':
        return {
            "text_features": tf.io.FixedLenFeature([4608], tf.float32),
            "image_features": tf.io.FixedLenFeature([data_config['mm_dims']['image']], tf.float32),
            "id_feature": tf.io.FixedLenFeature([id_fields_num], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32),
            "domain": tf.io.FixedLenFeature([1], tf.float32),
            "user_seq": tf.io.FixedLenFeature([seq_len], tf.int64),
        }
    else:  # tiktok, microlens
        return {
            "id_feature": tf.io.FixedLenFeature([id_fields_num], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32),
            "user_seq": tf.io.FixedLenFeature([seq_len], tf.int64),
        }


def stat_one_dataset(dataset_name, data_config, use_pkl=False):
    """统计单个数据集的所有信息"""
    tfrecord_path = data_config['data_dir']
    if not os.path.isabs(tfrecord_path):
        tfrecord_path = os.path.join(ROOT_DIR, tfrecord_path)

    id_fields_num = data_config['id_fields_num']

    all_users = set()
    all_items = set()
    split_stats = {}

    for split in ['train', 'val', 'test']:
        if use_pkl:
            # pkl 文件命名: {split}_{dataset_name}.pkl
            pkl_file = os.path.join(tfrecord_path, f'{split}_{dataset_name}.pkl')
            count, users, items, pos = stat_pkl(pkl_file)
        else:
            description = get_tfrecord_description(dataset_name, data_config)
            count, users, items, pos = stat_tfrecord(
                tfrecord_path, description, split, id_fields_num
            )

        split_stats[split] = {
            'count': count,
            'users': len(users),
            'items': len(items),
            'positive': pos,
        }
        all_users |= users
        all_items |= items

    # 多模态维度信息
    mm_dims = data_config.get('mm_dims', {})

    # 汇总
    total_count = sum(s['count'] for s in split_stats.values())
    total_positive = sum(s['positive'] for s in split_stats.values())

    return {
        'dataset': dataset_name,
        'splits': split_stats,
        'total_samples': total_count,
        'total_users': len(all_users),
        'total_items': len(all_items),
        'total_positive': total_positive,
        'total_negative': total_count - total_positive,
        'positive_ratio': total_positive / total_count if total_count > 0 else 0,
        'mm_dims': mm_dims,
        'seq_len': data_config.get('seq_len', 0),
        'id_feature_num': data_config.get('id_feature_num', 0),
    }


def print_report(stats):
    """打印格式化统计报告"""
    sep = '=' * 70
    print(sep)
    print(f"  数据集: {stats['dataset'].upper()}")
    print(sep)

    # 分割统计
    print(f"\n{'Split':<10} {'样本数':>12} {'正样本':>12} {'正样本率':>12} {'用户数':>12} {'物品数':>12}")
    print('-' * 70)
    for split in ['train', 'val', 'test']:
        s = stats['splits'][split]
        ratio = s['positive'] / s['count'] if s['count'] > 0 else 0
        print(f"{split:<10} {s['count']:>12,} {s['positive']:>12,} {ratio:>12.4f} {s['users']:>12,} {s['items']:>12,}")

    # 汇总
    print('-' * 70)
    print(f"{'Total':<10} {stats['total_samples']:>12,} {stats['total_positive']:>12,} "
          f"{stats['positive_ratio']:>12.4f} {stats['total_users']:>12,} {stats['total_items']:>12,}")

    # 多模态维度
    print(f"\n多模态特征维度:")
    for modal, dim in stats['mm_dims'].items():
        print(f"  {modal:<10}: {dim}D")
    print(f"  序列长度  : {stats['seq_len']}")
    print(f"  总特征数  : {stats['id_feature_num']}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='数据集统计')
    parser.add_argument('--datasets', nargs='+', default=['tiktok', 'antm2c', 'microlens'],
                        help='要统计的数据集名称')
    parser.add_argument('--use_pkl', action='store_true',default=False,
                        help='使用本地 pkl 文件统计 (注意: pkl仅含前5个batch, 结果不完整)')
    parser.add_argument('--config', default=None,
                        help='数据配置文件路径, 默认 config/data.yaml')
    args = parser.parse_args()

    config_path = args.config or os.path.join(ROOT_DIR, 'config', 'data.yaml')
    data_config = load_yaml(config_path)

    print("\n" + "=" * 70)
    print("  MMCTR Benchmark 数据集统计报告")
    if args.use_pkl:
        print("  [模式: 本地 pkl 文件 — 仅前5个batch, 统计不完整]")
    else:
        print("  [模式: TFRecord 全量数据]")
    print("=" * 70)

    all_stats = []
    for ds_name in args.datasets:
        if ds_name not in data_config:
            print(f"\n[警告] 数据集 '{ds_name}' 不在配置文件中, 跳过")
            continue
        print(f"\n正在统计 {ds_name} ...")
        stats = stat_one_dataset(ds_name, data_config[ds_name], use_pkl=args.use_pkl)
        all_stats.append(stats)
        print_report(stats)

    # 跨数据集对比汇总
    if len(all_stats) > 1:
        print("=" * 70)
        print("  跨数据集对比")
        print("=" * 70)
        print(f"\n{'Dataset':<12} {'Total':>12} {'Users':>12} {'Items':>12} {'Pos Ratio':>12} {'Modalities'}")
        print('-' * 80)
        for s in all_stats:
            modals = [k for k, v in s['mm_dims'].items() if k != 'id']
            modal_str = ', '.join(modals)
            print(f"{s['dataset']:<12} {s['total_samples']:>12,} {s['total_users']:>12,} "
                  f"{s['total_items']:>12,} {s['positive_ratio']:>12.4f} {modal_str}")
        print()


if __name__ == '__main__':
    main()
