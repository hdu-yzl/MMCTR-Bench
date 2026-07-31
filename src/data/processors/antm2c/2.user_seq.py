import pandas as pd
import os

if __name__ == '__main__':
    file_paths = ['data/processed/antm2c/train.csv', 'data/processed/antm2c/val.csv', 'data/processed/antm2c/test.csv']
    columns = [
        'user_id', 'item_id', 'original_item_id', 'log_time', 'label',
        'bill_entity_seq', 'service_entity_seq', 'query_entity_seq',
        'item_entity_names', 'item_title', 'scene',
    ]

    # 1. 读取并合并（保留全部样本）
    dfs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f'{fp} 不存在！')
        df = pd.read_csv(fp,
                         header=None,
                         names=columns,
                         usecols=['user_id', 'item_id', 'log_time', 'label'],
                         low_memory=False)
        df = df.iloc[1:]  # 去掉首行（如果首行是列名重复）
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data['log_time'] = pd.to_datetime(data['log_time'], errors='coerce')
    data = data.dropna(subset=['user_id', 'item_id', 'log_time'])

    user_seq = (data.sort_values(['user_id', 'log_time'])
                .groupby('user_id')
                .agg(item_seq=('item_id', lambda x: ','.join(x.astype(str))),
                     label_seq=('label', lambda x: ','.join(x.astype(str))))
                .reset_index())

    # 4. 保存
    csv_dir = 'data/processed/antm2c/'
    csv_path = os.path.join(csv_dir, "user_item_label_seq.csv")
    user_seq.to_csv(csv_path, index=False)
    print('已生成 user_item_label_seq.csv，共 {} 条用户序列'.format(len(user_seq)))
