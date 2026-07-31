import pandas as pd
import os
from collections import defaultdict
import random
import numpy as np

random.seed(2025)
np.random.seed(2025)

file_paths = [
    'data/raw/antm2c/antm2c_10m_part0',
    'data/raw/antm2c/antm2c_10m_part1',
    'data/raw/antm2c/antm2c_10m_part2'
]

columns = [
    'user_id', 'item_id', 'log_time', 'label', 'bill_entity_seq', 'service_entity_seq',
    'query_entity_seq', 'item_entity_names', 'item_title', 'scene',
    'deep_features_0', 'deep_features_1', 'deep_features_2', 'deep_features_3',
    'deep_features_4', 'deep_features_5', 'deep_features_6', 'deep_features_7',
    'deep_features_8', 'deep_features_9', 'deep_features_10', 'deep_features_11',
    'deep_features_12', 'deep_features_13', 'deep_features_14', 'deep_features_15',
    'deep_features_16', 'deep_features_17', 'deep_features_18', 'deep_features_19',
    'deep_features_20', 'deep_features_21', 'deep_features_22', 'deep_features_23',
    'deep_features_24', 'deep_features_25', 'deep_features_26'
]


def collect_feature_values(data, feature_columns):
    feature_values = defaultdict(set)
    for col in feature_columns:
        feature_values[col].update(data[col].dropna().unique())
    return feature_values


def create_feature_mapping(feature_values):
    feature_mapping = {}
    current_mapping_value = 1
    for col, values in feature_values.items():
        feature_mapping[col] = {value: current_mapping_value + i for i, value in enumerate(values)}
        current_mapping_value += len(values)
    return feature_mapping


def remap_features(data, feature_mapping, feature_columns):
    for col in feature_columns:
        data.loc[:, col] = data[col].map(feature_mapping[col])
    return data


def save_to_csv(data, filename):
    columns_to_save = [
        'user_id', 'item_id', 'original_item_id', 'log_time', 'label', 'bill_entity_seq', 'service_entity_seq',
        'query_entity_seq', 'item_entity_names', 'item_title', 'scene'
    ]
    data[columns_to_save].to_csv(filename, index=False)


if __name__ == '__main__':
    train_data_list = []
    val_data_list = []
    test_data_list = []

    all_data = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, names=columns, header=None, low_memory=False)[1:]
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)

    feature_columns = ['user_id', 'item_id']
    feature_values = collect_feature_values(all_data, feature_columns)

    feature_mapping = create_feature_mapping(feature_values)

    for file_path in file_paths:
        data = pd.read_csv(file_path, names=columns, header=None, low_memory=False)[1:]
        data['log_time'] = pd.to_datetime(data['log_time'],
                                          format='%Y-%m-%d %H:%M:%S',
                                          errors='coerce')
        default_time = pd.Timestamp('2023-08-01 12:00:00')
        data['log_time'] = data['log_time'].fillna(default_time)

        train_data = data[data['log_time'] <= '2023-08-03']
        val_data = data[(data['log_time'] > '2023-08-03') & (data['log_time'] <= '2023-08-05')]
        test_data = data[data['log_time'] > '2023-08-05']

        # 添加原始 item_id 列
        train_data.loc[:, 'original_item_id'] = train_data['item_id']
        val_data.loc[:, 'original_item_id'] = val_data['item_id']
        test_data.loc[:, 'original_item_id'] = test_data['item_id']

        train_data = remap_features(train_data, feature_mapping, feature_columns)
        val_data = remap_features(val_data, feature_mapping, feature_columns)
        test_data = remap_features(test_data, feature_mapping, feature_columns)

        train_data_list.append(train_data)
        val_data_list.append(val_data)
        test_data_list.append(test_data)

    train_data = pd.concat(train_data_list, ignore_index=True)
    val_data = pd.concat(val_data_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)

    save_to_csv(train_data, 'data/processed/antm2c/train.csv')
    save_to_csv(val_data, 'data/processed/antm2c/val.csv')
    save_to_csv(test_data, 'data/processed/antm2c/test.csv')
