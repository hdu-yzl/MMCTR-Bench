import numpy as np
import pandas as pd


# 构建 item 特征矩阵, item_id -> emb_vec ,0: padding
def build_emb_matrix(df, id_col, emb_col, emb_dim, max_item_id):
    if max_item_id is None:
        max_item_id = int(df[id_col].max())

    mat = np.zeros((max_item_id + 1, emb_dim), dtype=np.float32)

    for _, row in df.iterrows():
        if _ == 0:
            continue
        idx = int(row[id_col])
        vec = np.asarray(row[emb_col], dtype=np.float32)
        mat[idx] = vec
    return mat


def main():
    path = 'data/raw/microlens/item_feature.parquet'
    df = pd.read_parquet(path)

    max_item_id = int(df["item_id"].max())  # 91717
    min_item_id = 1
    print(f"max_item_id = {max_item_id}")

    text_mat = build_emb_matrix(df, "item_id", "txt_emb_BERT", 768, max_item_id)
    img_mat = build_emb_matrix(df, "item_id", "img_emb_CLIPRN50", 1024, max_item_id)

    text_out = 'data/processed/microlens/text_emb.npy'
    img_out = 'data/processed/microlens/img_emb.npy'

    np.save(text_out, text_mat)
    np.save(img_out, img_mat)
    print(f"saved: {text_out}  shape={text_mat.shape}")
    print(f"saved: {img_out}  shape={img_mat.shape}")


if __name__ == "__main__":
    main()
