import pandas as pd
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import subprocess

file_paths = [
    'data/processed/antm2c/train.csv',
    'data/processed/antm2c/val.csv',
    'data/processed/antm2c/test.csv'
]
picture_path = 'data/raw/antm2c/AntM2C_image'
columns = [
    'user_id', 'item_id', 'original_item_id','log_time', 'label', 'bill_entity_seq', 'service_entity_seq',
        'query_entity_seq', 'item_entity_names', 'item_title', 'scene','prev_5_items'
]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

clip_model = ChineseCLIPModel.from_pretrained("data/pakage/clip").to(device)
clip_model.eval()
preprocess = ChineseCLIPProcessor.from_pretrained("data/pakage/clip")


def extract_image_features_batch(item_ids, original_item_ids):
    image_features = []
    for i in tqdm(range(0, len(item_ids)), desc="Extracting image features"):
        original_id = original_item_ids[item_ids[i]]
        image_path = os.path.join(picture_path, f'{original_id}.png')  # 使用 original_item_id 作为图片名
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"Missing image for original_item_id: {original_id}")
            image_features.append(np.zeros(512))
            continue

        with torch.no_grad():
            inputs = preprocess(images=image, return_tensors="pt").to(device)
            outputs = clip_model.get_image_features(**inputs)
            image_features.append(outputs.detach().cpu().numpy().flatten())

    return np.vstack(image_features).astype(np.float32)


def get_item_id_to_original_id_mapping(data):
    item_to_original = {}
    for _, row in data.iterrows():
        item_id = row['item_id']
        original_item_id = row['original_item_id']
        if item_id not in item_to_original:
            item_to_original[item_id] = original_item_id
    return item_to_original


if __name__ == '__main__':
    train_data = pd.read_csv(file_paths[0], low_memory=False)
    val_data = pd.read_csv(file_paths[1], low_memory=False)
    test_data = pd.read_csv(file_paths[2], low_memory=False)

    all_data = pd.concat([train_data, val_data, test_data])

    # 获取 item_id 到 original_item_id 的一对一映射关系，用于找原始图片
    item_to_original_mapping = get_item_id_to_original_id_mapping(all_data)

    unique_item_ids = sorted(item_to_original_mapping.keys())
    print(f"Total unique item_ids: {len(unique_item_ids)}")

    image_features = extract_image_features_batch(unique_item_ids, item_to_original_mapping)

    min_item_id = min(unique_item_ids)
    max_item_id = max(unique_item_ids)
    print(f"Minimum item_id: {min_item_id}, Maximum item_id: {max_item_id}")

    feature_matrix = np.zeros((max_item_id - min_item_id + 2, image_features.shape[1]), dtype=np.float32)

    for idx, item_id in enumerate(unique_item_ids):
        feature_matrix[item_id - min_item_id+1] = image_features[idx]

    print(f"Feature matrix shape: {feature_matrix.shape}")
    np.save('data/processed/antm2c/image_feature.npy', feature_matrix)
