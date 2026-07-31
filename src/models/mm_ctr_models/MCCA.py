from pathlib import Path
from ..base_seq_model import BaseSeqModel
from ..layers import seq_pooling
import torch
from ..layers.common import MultiLayerPerceptron, FeatureEmbedding
from models.pre_models.PSRQ import PSRQ_Premodel as PSRQ


class MCCA(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        # 优先使用数据集特定配置，缺省时回退到顶层配置（与其他模型保持一致）
        model_config = model_config.get(data_config['name'], model_config)

        # 推荐模型 checkpoint 文件名加上数据集名，避免不同数据集互相覆盖
        self.ckpt_path = Path(self.train_config['checkpoint_dir']) / \
            f"{self.data_config['name']}_{self.model_config.get('model_name', 'mcca')}.pt"
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.n_levels = model_config.get('n_levels', 3)
        self.codebook_size = model_config.get('codebook_size', 1024)
        self.num_emb_list = [self.codebook_size for _ in range(self.n_levels)]

        self.rq_features = [k for k in self.mm_features if k != 'id']

        self.PSRQ = PSRQ(model_config, train_config, data_config)
        self.PSRQ.load()
        # 确保 PSRQ 模型在正确的设备上
        self.PSRQ.to(device=self.device)
        self.PSRQ.train(False)

        self.modal_embeddings = torch.nn.ModuleDict({
            m: torch.nn.ModuleList([
                FeatureEmbedding(self.num_emb_list[0], self.latent_dim)
                for _ in range(self.n_levels)
            ])
            for m in self.rq_features
        })
        self.joint_embedding = torch.nn.ModuleList([
            FeatureEmbedding(self.num_emb_list[0], self.latent_dim)
            for _ in range(self.n_levels)
        ])

        # 重写模态维度
        for k in self.rq_features:
            self.mm_dims[k] = int(self.latent_dim * self.n_levels)

        self.mm_projector = torch.nn.ModuleDict({k: torch.nn.Linear(self.mm_dims[k], self.projection_dim)
                                                 for k in self.mm_features})
        self.joint_projector = torch.nn.Linear(int(self.latent_dim * self.n_levels), self.projection_dim)

        self.model_attns_pooling = torch.nn.ModuleDict({
            m: seq_pooling.get_pooling('cross_atten', self.projection_dim)
            for m in self.mm_features  # 包含id
        })
        self.input_dim = self.projection_dim * (self.mm_nums + self.user_features_num + 1)  # 1是用户id
        self.dnn = MultiLayerPerceptron(
            self.input_dim,
            self.mlp_dims, self.dropout,
            use_bn=self.bn)

        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def model_to_device(self):
        """重写model_to_device以同时移动PSRQ"""
        super().model_to_device()  # 移动MCCA自身的模块
        # 同时移动PSRQ到相同设备
        self.PSRQ.to(device=self.device)

    def psrq_encode(self, feats, feats_seq):
        feats_id, joint_id, feats_seq_id = self.PSRQ.get_encode(feats, feats_seq)
        return feats_id, joint_id, feats_seq_id

    def get_alignment_feats(self, feats, feats_seq):
        """为模态对齐实验构造每模态表征。

        流程：获取码本中最近的向量（PSRQ 编码）→ 多层码本嵌入查表 →
        投影对齐 → 序列侧 mean pooling，得到每模态 (B, projection_dim) 表征，
        供对齐损失将非 ID 模态与 ID 模态对齐。
        """
        feats = dict(feats)
        feats_seq = dict(feats_seq)

        feats['id'] = self.embedding(feats['id']).squeeze()
        feats_seq['id'] = self.embedding(feats_seq['id'])

        _, _, feats_seq_id = self.psrq_encode(feats, feats_seq)
        for m in self.rq_features:
            seq_emb = []
            for i in range(self.n_levels):
                seq_emb.append(self.modal_embeddings[m][i](feats_seq_id[m][:, :, i]))
            feats_seq[m] = torch.cat(seq_emb, dim=-1)  # (B, seq_num, n_levels*latent_dim)

        align_feats = {}
        for m in self.mm_features:
            seq_proj = self.mm_projector[m](feats_seq[m])  # (B, seq_num, projection_dim)
            align_feats[m] = seq_proj.mean(dim=1)          # (B, projection_dim)
        return align_feats

    def forward(self, user_feats, feats, feats_seq):
        # 鲁棒性实验：在 PSRQ 编码前记录被整体置零的缺失模态样本。
        # （PSRQ 把零向量也会映射到某个确定码字，故需显式识别缺失样本，
        #   并在投影前同步置零，使模态缺失与连续模态模型语义一致。）
        modal_keep = self.build_modal_keep_mask(feats) if self.modal_drop_enabled else None

        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) userid,itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_id, joint_id, feats_seq_id = self.psrq_encode(feats, feats_seq)

        for m in self.rq_features:
            emb = []
            seq_emb = []
            for i in range(self.n_levels):
                # 使用序列索引 (B, seq_num, n_levels) -> 第i个level的索引
                seq_emb.append(self.modal_embeddings[m][i](feats_seq_id[m][:, :, i]))
                # 使用单项索引 (B, n_levels) -> 第i个level的索引
                emb.append(self.modal_embeddings[m][i](feats_id[m][:, i]))
            feats[m] = torch.cat(emb, dim=-1)  # (B,n_layers*latent_dim)
            feats_seq[m] = torch.cat(seq_emb, dim=-1)  # (B,seq_num,n_layers*latent_dim)

        # 鲁棒性实验：将缺失模态的离散编码表征在投影前置零。
        if modal_keep is not None:
            for m in self.rq_features:
                keep = modal_keep[m]
                feats[m] = feats[m] * keep.unsqueeze(-1)
                feats_seq[m] = feats_seq[m] * keep.view(-1, 1, 1)

        emb = []
        for i in range(self.n_levels):
            emb.append(self.joint_embedding[i](joint_id[:, i]))

        joint_id = torch.cat(emb, dim=-1)  # (B,n_layers*latent_dim)

        # 鲁棒性实验：joint 表征由各模态原始特征拼接量化而来；
        # 当样本所有非 ID 模态均缺失时，joint 也应视为缺失并置零。
        if modal_keep is not None:
            joint_keep = torch.clamp(
                sum(modal_keep[m] for m in self.rq_features), max=1.0)
            joint_id = joint_id * joint_keep.unsqueeze(-1)

        feats_p, feats_seq_p = {}, {}
        for m in self.mm_features:
            feats_p[m] = self.mm_projector[m](feats[m])
            feats_seq_p[m] = self.mm_projector[m](feats_seq[m])
        joint_id_p = self.joint_projector(joint_id)

        feats_pooling = {}
        for m in self.mm_features:
            if m == 'id':
                feats_pooling[m] = self.model_attns_pooling[m](feats_p[m], feats_seq_p[m], feats_seq_p[m])
            else:
                feats_pooling[m] = self.model_attns_pooling[m](joint_id_p, feats_seq_p[m],
                                                               feats_seq_p[m])  # (B,projection_dim)

        feats_tensors = [feats_pooling[k] for k in feats_pooling]
        feats_vec = torch.cat(feats_tensors, dim=-1)  # (B,mm_field_num*projection_dim)

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=-1)  # (B,user_field_num*projection_dim)

        x_dnn = torch.cat([user_vec, feats_vec, joint_id_p],
                          dim=-1)  # (B,(mm_field_num+user_field_num+1)*projection_dim)
        dnn_out = self.dnn(x_dnn)
        logit = self.out_put(dnn_out)
        out = {'pred': logit}
        return out
