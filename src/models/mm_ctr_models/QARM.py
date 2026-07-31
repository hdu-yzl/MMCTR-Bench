from pathlib import Path
from ..base_seq_model import BaseSeqModel
from ..layers import seq_pooling
import torch
import numpy as np
from ..layers.common import MultiLayerPerceptron, FeatureEmbedding, CrossNetwork
from models.pre_models.RQ import ResidualQuantizer as RQ


class QARM(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        # 优先使用数据集特定配置，缺省时回退到顶层配置（与其他模型保持一致）
        model_config = model_config.get(data_config['name'], model_config)
        print(f"Initializing QARM with config: {model_config}")

        # 推荐模型 checkpoint 文件名加上数据集名，避免不同数据集互相覆盖
        self.ckpt_path = Path(self.train_config['checkpoint_dir']) / \
            f"{self.data_config['name']}_{self.model_config.get('model_name', 'qarm')}.pt"
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.codebook_size = model_config.get('codebook_size', 1024)
        self.n_levels = model_config.get('n_levels', 3)
        self.cross_num = model_config.get('cross_num', 3)
        self.rq_features = [k for k in self.mm_features if k != 'id']
        self.input_dim = self.projection_dim * (self.mm_nums * 2 + self.user_features_num)  # 1是用户id

        self.modal_embeddings = torch.nn.ModuleDict({
            m: torch.nn.ModuleList([
                FeatureEmbedding(self.codebook_size, self.latent_dim)
                for _ in range(self.n_levels)
            ])
            for m in self.rq_features
        })

        # RQ 使用字典存储，但需要手动管理设备
        self.rqs = {
            m: RQ(model_config, train_config, data_config)
            for m in self.rq_features
        }

        # 重写模态维度
        for k in self.rq_features:
            self.mm_dims[k] = int(self.latent_dim * self.n_levels)

        self.mm_projector = torch.nn.ModuleDict({k: torch.nn.Linear(self.mm_dims[k], self.projection_dim)
                                                 for k in self.mm_features})

        self.pooling = seq_pooling.get_pooling('mean')

        self.cross = CrossNetwork(self.input_dim, self.cross_num)
        self.dnn = MultiLayerPerceptron(
            self.input_dim,
            self.mlp_dims, self.dropout,
            use_bn=self.bn)
        self.comb = MultiLayerPerceptron(
            self.input_dim + self.mlp_dims[-1],
            self.mlp_dims, self.dropout,
            use_bn=self.bn)

        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.log_model_params()
        self.load_rq()
        self.model_to_device()
        self.rq_to_device()

    def load_rq(self):
        """加载预训练的RQ codebooks（统一从 checkpoint_dir 读取）"""
        for m in self.rq_features:
            rq_path = f"{self.train_config['checkpoint_dir']}/{self.data_config['name']}_{m}_rq"
            self.rqs[m].load(rq_path)

    def rq_to_device(self):
        """将RQ codebooks移到指定设备"""
        for m in self.rq_features:
            self.rqs[m].codebooks = [
                torch.from_numpy(cb).to(self.device) if isinstance(cb, np.ndarray) else cb.to(self.device)
                for cb in self.rqs[m].codebooks
            ]

    def rq_encode(self, feats, feats_seq):
        """对特征进行RQ编码，需要先进行L2 normalize"""
        epsilon = 1e-8
        for m in self.rq_features:
            # L2 normalize (保持与RQ训练时一致)
            norm_seq = feats_seq[m] / torch.clamp(torch.norm(feats_seq[m], dim=-1, keepdim=True), min=epsilon)
            norm_feat = feats[m] / torch.clamp(torch.norm(feats[m], dim=-1, keepdim=True), min=epsilon)
            
            codes, _ = self.rqs[m].encode_tensor(norm_seq)
            feats_seq[m] = codes
            codes, _ = self.rqs[m].encode_tensor(norm_feat)
            feats[m] = codes

    def get_alignment_feats(self, feats, feats_seq):
        """为模态对齐实验构造每模态表征。

        流程：获取码本中最近的向量（RQ 编码）→ 多层码本嵌入查表 →
        投影对齐 → 序列侧 mean pooling，得到每模态 (B, projection_dim) 表征，
        供对齐损失将非 ID 模态与 ID 模态对齐。
        """
        feats = dict(feats)
        feats_seq = dict(feats_seq)

        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        self.rq_encode(feats, feats_seq)
        for m in self.rq_features:
            seq_emb = []
            for i in range(self.n_levels):
                seq_emb.append(self.modal_embeddings[m][i](feats_seq[m][:, :, i]))
            feats_seq[m] = torch.cat(seq_emb, dim=-1)  # (B, seq_num, n_levels*latent_dim)

        align_feats = {}
        for m in self.mm_features:
            seq_proj = self.mm_projector[m](feats_seq[m])  # (B, seq_num, projection_dim)
            align_feats[m] = self.pooling(seq_proj)        # (B, projection_dim)
        return align_feats

    def forward(self, user_feats, feats, feats_seq):
        # 鲁棒性实验：在 RQ 编码改写 feats 之前，先记录被整体置零的缺失模态样本。
        # （RQ 把零向量也会映射到某个确定码字，故需在此显式识别缺失样本，
        #   并在投影前同步置零，使模态缺失与连续模态模型语义一致。）
        modal_keep = self.build_modal_keep_mask(feats) if self.modal_drop_enabled else None

        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) userid,itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        self.rq_encode(feats, feats_seq)
        for m in self.rq_features:
            emb = []
            seq_emb = []
            for i in range(self.n_levels):
                seq_emb.append(self.modal_embeddings[m][i](feats_seq[m][:, :, i]))
                emb.append(self.modal_embeddings[m][i](feats[m][:, i]))
            feats[m] = torch.cat(emb, dim=-1)  # (B,n_levels*latent_dim)
            feats_seq[m] = torch.cat(seq_emb, dim=-1)  # (B,seq_num,n_levels*latent_dim)

        # 鲁棒性实验：将缺失模态的离散编码表征在投影前置零，
        # 使其投影后退化为常量（bias），与连续模态模型置零原始输入的语义对齐。
        if modal_keep is not None:
            for m in self.rq_features:
                keep = modal_keep[m]
                feats[m] = feats[m] * keep.unsqueeze(-1)
                feats_seq[m] = feats_seq[m] * keep.view(-1, 1, 1)

        for m in self.mm_features:
            feats[m] = self.mm_projector[m](feats[m])
            feats_seq[m] = self.mm_projector[m](feats_seq[m])
            feats_seq[m] = self.pooling(feats_seq[m])

        feats_tensors = [feats[k] for k in feats]
        feats_vec = torch.cat(feats_tensors, dim=-1)  # (B,mm_field_num*projection_dim)

        feats_seq_tensors = [feats_seq[k] for k in feats_seq]
        feats_seq_vec = torch.cat(feats_seq_tensors, dim=-1)  # (B,mm_field_num*projection_dim)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=-1)  # (B,user_field_num*projection_dim)

        x_dnn = torch.cat([user_vec, feats_vec, feats_seq_vec],
                          dim=-1)  # (B,(mm_field_num*2+user_field_num)*projection_dim)
        cross_out = self.cross(x_dnn)
        dnn_out = self.dnn(x_dnn)

        x_dnn = torch.cat((cross_out, dnn_out), dim=-1)

        dnn_out = self.comb(x_dnn)
        logit = self.out_put(dnn_out)
        out = {'pred': logit}
        return out
