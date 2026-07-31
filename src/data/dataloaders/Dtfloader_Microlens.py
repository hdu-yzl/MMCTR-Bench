import numpy as np
import tensorflow as tf
import glob
import torch
import pickle
import os


class MicrolensLoader(object):
    def __init__(self, data_config, batch_size):
        self.bs = batch_size
        self.samples = 1
        self.id_features = data_config['id_fields_num']
        self.text_features = 768
        self.image_features = 1024
        self.seq_len = data_config['seq_len']  # 已左padding 0
        self.tfrecord_path = data_config['data_dir']

        self.min_item_id = 1000001
        self.min_user_id = 1

        self.description = {
            "id_feature": tf.io.FixedLenFeature([self.id_features], tf.int64),
            "label": tf.io.FixedLenFeature([self.samples], tf.float32),
            'user_seq': tf.io.FixedLenFeature([self.seq_len], tf.int64),
        }

        self.image_feature = np.load(self.tfrecord_path + '/' + 'img_emb.npy')  # (item_num+1,4096)
        self.title_feature = np.load(self.tfrecord_path + '/' + 'text_emb.npy')  # (item_num+1,384)

        self.title_feat_torch = torch.from_numpy(self.title_feature)
        self.image_feat_torch = torch.from_numpy(self.image_feature)

        self.saved_batches_dir = os.path.join(self.tfrecord_path, 'saved_batches')

    def get_multi_modal(self, ):
        return {'text': self.title_feat_torch.numpy(), 'image': self.image_feat_torch.numpy()}

    def _save_batches(self, batches, filename):
        path = os.path.join(self.saved_batches_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(batches, f)
        print(f"Saved {len(batches)} batches to {path}")

    def _load_batches(self, filename):
        path = os.path.join(self.tfrecord_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saved batch file not found: {path}")
        with open(path, 'rb') as f:
            batches = pickle.load(f)
            # print(f"Loaded {len(batches)} batches from {path}")

        return batches

    def save_first_n_batches(self, data_type='test'):
        # print("Saving first 5 batches from get_data...")
        batches_data = []
        data_gen = self.get_data(data_type)
        for i, batch in enumerate(data_gen):
            if i >= 5:
                break
            batches_data.append(batch)
        self._save_batches(batches_data, f'{data_type}_microlens.pkl')

    def save_first_n_seq_batches(self, data_type='test'):
        # print("Saving first 5 batches from get_seq_data...")
        batches_seq = []
        seq_gen = self.get_data_seq(data_type)
        for i, batch in enumerate(seq_gen):
            if i >= 5:
                break
            batches_seq.append(batch)
        self._save_batches(batches_seq, f'{data_type}_microlens_seq.pkl')

    def get_data(self, data_type):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return (
                example['id_feature'],
                example['label'],
                example['user_seq']
            )

        files = glob.glob(self.tfrecord_path + '/' + "{}*.tfrecord".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(self.bs).prefetch(tf.data.experimental.AUTOTUNE)
        for id_feat, label, user_seq in ds:
            user_seq = torch.from_numpy(user_seq.numpy())
            id_feat = torch.from_numpy(id_feat.numpy())
            label = torch.from_numpy(label.numpy())

            item_id = id_feat[:, 1]
            item_text = self.title_feat_torch[item_id - self.min_item_id + 1]
            item_image = self.image_feat_torch[item_id - self.min_item_id + 1]

            idx = torch.where(user_seq == 0, 0, user_seq - self.min_item_id + 1)
            text_seq = self.title_feat_torch[idx]
            image_seq = self.image_feat_torch[idx]

            feats = {'id': id_feat, 'text': item_text, 'image': item_image}
            feats_seq = {'id': user_seq, 'text': text_seq, 'image': image_seq}

            yield feats, feats_seq, label

    def get_data_seq(self, data_type):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return (
                example['id_feature'],
                example['label'],
                example['user_seq']
            )

        files = glob.glob(self.tfrecord_path + '/' + "{}*.tfrecord".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(self.bs).prefetch(tf.data.experimental.AUTOTUNE)
        for id_feat, label, user_seq in ds:
            id_feat = torch.from_numpy(id_feat.numpy())
            label = torch.from_numpy(label.numpy())
            user_seq = torch.from_numpy(user_seq.numpy())

            item_id = id_feat[:, 1]
            item_text = self.title_feat_torch[item_id - self.min_item_id + 1]
            item_image = self.image_feat_torch[item_id - self.min_item_id + 1]

            idx = torch.where(user_seq == 0, 0, user_seq - self.min_item_id + 1)
            text_seq = self.title_feat_torch[idx]
            image_seq = self.image_feat_torch[idx]

            user_feats = {'id': id_feat[:, 0:1]}
            feats = {'id': id_feat[:, 1:2], 'text': item_text, 'image': item_image}
            feats_seq = {'id': user_seq, 'text': text_seq, 'image': image_seq}

            yield user_feats, feats, feats_seq, label
