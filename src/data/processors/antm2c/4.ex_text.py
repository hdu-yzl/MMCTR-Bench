import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import os

# 确保保存目录存在
os.makedirs('data/processed/antm2c', exist_ok=True)

file_paths = [
    'data/processed/antm2c/train.csv',
    'data/processed/antm2c/val.csv',
    'data/processed/antm2c/test.csv'
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained('data/pakage/bert')

def process_sequences(sequences, device, batch_size=256, save_path=None):
    model = BertModel.from_pretrained('data/pakage/bert').to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Processing on {device}"):
            batch = sequences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            pooled_output = outputs.pooler_output.cpu().numpy()
            embeddings.append(pooled_output)
    embeddings = np.concatenate(embeddings, axis=0)
    if save_path:
        np.save(save_path, embeddings)
    return embeddings

def process_item_title_dedup():
    print("Processing deduplicated item_title embeddings...")

    # 读取所有数据并合并
    all_data = []
    for fp in file_paths:
        df = pd.read_csv(fp, usecols=['item_id', 'item_title'], low_memory=False)
        all_data.append(df)
    full_df = pd.concat(all_data, ignore_index=True)

    # 去除 item_title 为空的行
    full_df = full_df.dropna(subset=['item_title'])
    full_df['item_title'] = full_df['item_title'].astype(str)

    # 按 item_id 去重，保留第一个出现的 title
    dedup_df = full_df.drop_duplicates(subset=['item_id'], keep='first')

    # 获取全局最大 item_id
    max_item_id = dedup_df['item_id'].max()
    min_item_id = dedup_df['item_id'].min()
    print(f"Max item_id: {max_item_id}")

    # 准备 item_id 到 title 的映射
    item_ids = dedup_df['item_id'].tolist()
    titles = dedup_df['item_title'].tolist()

    # 加载模型
    model = BertModel.from_pretrained('data/pakage/bert').to(device)
    model.eval()

    # 编码 titles
    batch_size = 256
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(titles), batch_size), desc="Encoding item_title"):
            batch_titles = titles[i:i + batch_size]
            inputs = tokenizer(batch_titles, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            pooled = outputs.pooler_output.cpu().numpy()
            embeddings.append(pooled)
    embeddings = np.concatenate(embeddings, axis=0)  # shape: (N_unique, 768)

    # 创建 (max_item_id + 1, 768) 的矩阵，初始化为 0
    feature_matrix = np.zeros((max_item_id-min_item_id + 2, embeddings.shape[1]), dtype=np.float32)

    # 将 embedding 填入对应 item_id 行
    for idx, item_id in enumerate(item_ids):
        feature_matrix[item_id-min_item_id+1] = embeddings[idx]

    # 保存
    save_path = 'data/processed/antm2c/text_feature.npy'
    np.save(save_path, feature_matrix)
    print(f"Saved item_title embeddings to {save_path} with shape {feature_matrix.shape}")

if __name__ == '__main__':
    columns_to_use = ['bill_entity_seq', 'service_entity_seq',
                      'query_entity_seq', 'item_entity_names', 'log_time']

    # 处理普通列（保持原逻辑）
    for column in columns_to_use:
        train_data = pd.read_csv(file_paths[0], low_memory=False)
        seq1 = train_data[column].fillna('').tolist()
        process_sequences(seq1, device, save_path=f'data/processed/antm2c/{column}_embeddings_train.npy')

        val_data = pd.read_csv(file_paths[1], low_memory=False)
        seq2 = val_data[column].fillna('').tolist()
        process_sequences(seq2, device, save_path=f'data/processed/antm2c/{column}_embeddings_val.npy')

        test_data = pd.read_csv(file_paths[2], low_memory=False)
        seq3 = test_data[column].fillna('').tolist()
        process_sequences(seq3, device, save_path=f'data/processed/antm2c/{column}_embeddings_test.npy')

    # 特殊处理 item_title
    process_item_title_dedup()

    print("All sequences processed and embeddings saved.")