import pandas as pd
from tqdm import tqdm
import os
if __name__ == '__main__':

    tqdm.pandas()
    columns = [
        'user_id', 'item_id', 'original_item_id', 'log_time', 'label',
        'bill_entity_seq', 'service_entity_seq', 'query_entity_seq',
        'item_entity_names', 'item_title', 'scene',
    ]
    # 读取用户序列数据
    seq_df = pd.read_csv('data/processed/antm2c/user_item_label_seq.csv', header=None,
                         names=['user_id', 'item_seq', 'label_seq']).iloc[1:]

    user_dict = {}
    for _, r in seq_df.iterrows():
        uid = str(r['user_id'])
        items = str(r['item_seq']).split(',')
        labels = str(r['label_seq']).split(',')
        user_dict[uid] = (items, labels)

    file_paths = ['data/processed/antm2c/train.csv', 'data/processed/antm2c/val.csv', 'data/processed/antm2c/test.csv']

    for fp in file_paths:
        print(f'Processing {fp} ...')
        df = pd.read_csv(fp, names=columns, header=None, low_memory=False).iloc[1:]

        def get_prev_5_clicked(row):
            uid = str(row['user_id'])
            tid = str(row['item_id'])
            if uid not in user_dict:
                return '0 0 0 0 0'
            items, labels = user_dict[uid]

            try:
                # 找到当前 item 的位置
                pos = items.index(tid)
            except ValueError:
                return '0 0 0 0 0'

            # 只取当前物品之前且 label=='1' 的 item
            clicked = [items[i] for i in range(pos) if labels[i] == '1']
            # 取最后 5 个
            prev = clicked[-5:]
            # 左补 0
            prev = ['0'] * (5 - len(prev)) + prev
            return ' '.join(prev)


        df['prev_5_items'] = df.progress_apply(get_prev_5_clicked, axis=1)


        df.to_csv(fp, index=False)
        print(f'Saved {fp}')
