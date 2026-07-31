import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm

file_paths = {
    'train': 'data/processed/antm2c/train.csv',
    'val': 'data/processed/antm2c/val.csv',
    'test': 'data/processed/antm2c/test.csv'
}
columns = [
    'user_id', 'item_id', 'original_item_id', 'log_time', 'label', 'bill_entity_seq', 'service_entity_seq',
    'query_entity_seq', 'item_entity_names', 'item_title', 'scene', 'prev_5_items'
]


def merge_id_features(row, feature_columns):
    id_features = []
    for col in feature_columns:
        id_features.append(row[col])
    return id_features


def create_example(text_features, image_features, id_features, label, scene, user_seq):
    feature = {
        'text_features': tf.train.Feature(float_list=tf.train.FloatList(value=text_features)),
        'image_features': tf.train.Feature(float_list=tf.train.FloatList(value=image_features)),
        'id_feature': tf.train.Feature(int64_list=tf.train.Int64List(value=id_features)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'domain': tf.train.Feature(float_list=tf.train.FloatList(value=[scene])),
        'user_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=user_seq))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord_train(data, data_type, id_feature_columns, image_feature, text_feature):
    writer = tf.io.TFRecordWriter(f'data/{data_type}.tfrecord')
    data['user_seq'] = (
        data['prev_5_items']
            .astype(str)
            .str.split()
            .apply(lambda x: [int(i) for i in x])
    )
    for i, row in tqdm(data.iterrows(), total=len(data)):
        id_features = merge_id_features(row, id_feature_columns)
        label = float(row['label'])
        scene = float(row['scene'])
        user_seq = row['user_seq']
        text_features = np.concatenate(
            (service_embeddings_train[i], query_embeddings_train[i], bill_embeddings_train[i],
             item_entity_names_embeddings_train[i], text_feature[id_features[1] - Min_item_id +1],
             log_time_embeddings_train[i]))

        example = create_example(text_features, image_feature[id_features[1] - Min_item_id +1], id_features, label, scene, user_seq)

        writer.write(example)
    writer.close()



if __name__ == '__main__':
    id_feature_columns = ['user_id', 'item_id']

    image_features = np.load('data/processed/antm2c/image_feature.npy')
    text_features = np.load('data/processed/antm2c/text_feature.npy')
    print('done loading image features, shape:', image_features.shape)
    Min_item_id = 67625

    # train_data
    service_embeddings_train = np.load('data/processed/antm2c/service_embeddings_train.npy')
    print('done loading service embeddings, shape:', service_embeddings_train.shape)

    query_embeddings_train = np.load('data/processed/antm2c/query_embeddings_train.npy')
    print('done loading query embeddings, shape:', query_embeddings_train.shape)

    bill_embeddings_train = np.load('data/processed/antm2c/bill_embeddings_train.npy')
    print('done loading bill embeddings, shape:', bill_embeddings_train.shape)

    item_entity_names_embeddings_train = np.load('data/processed/antm2c/item_entity_names_embeddings_train.npy')
    print('done loading item_entity_names embeddings, shape:', item_entity_names_embeddings_train.shape)

    log_time_embeddings_train = np.load('data/processed/antm2c/log_time_embeddings_train.npy')
    print('done loading log_time embeddings, shape:', log_time_embeddings_train.shape)

    for data_type, data_path in file_paths.items():
        data = pd.read_csv(data_path, low_memory=False)
        print(f'{data_type} data shape:', data.shape)

        write_tfrecord_train(data, data_type, id_feature_columns, image_features, text_features)
