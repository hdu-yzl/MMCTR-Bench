import pandas as pd
import random

random.seed(42)
max_user_id = 9319

df = pd.read_csv('data/processed/tiktok/ctr_dataset.csv')


def shift_item(x):
    return x + 1 if x == -1 else x + max_user_id + 1


def shift_seq(seq_str):
    return ','.join(
        str(int(t) + 1) if int(t) == -1 else str(int(t) + max_user_id + 1)
        for t in seq_str.split(',')
    )


df['itemid'] = df['itemid'].apply(shift_item)  # 9320 - 9320+6710
df['userid'] = df['userid'] + 1  # 1-9319
df['userseq'] = df['userseq'].apply(shift_seq)

# ---------- 3. 8:1:1 随机划分 ----------
n = len(df)
idx = list(range(n))
random.shuffle(idx)

train_idx = idx[:int(0.8 * n)]
val_idx = idx[int(0.8 * n):int(0.9 * n)]
test_idx = idx[int(0.9 * n):]

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# ---------- 4. 保存 ----------
train_df.to_csv('data/processed/tiktok/train.csv', index=False)
val_df.to_csv('data/processed/tiktok/val.csv', index=False)
test_df.to_csv('data/processed/tiktok/test.csv', index=False)

print('划分完成：')
print('train:', len(train_df))
print('val  :', len(val_df))
print('test :', len(test_df))
