import torch
from models.layers.common import MultiLayerPerceptron, FeatureEmbedding
import torch.nn as nn
import torch.nn.functional as F
import random
from mmctr.utils import helper
import numpy as np
from pathlib import Path


class PSRQ_Premodel(torch.nn.Module):
    def __init__(self,
                 model_config,
                 train_config,
                 data_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.set_up()

        self.device = helper.getDevice(self.train_config["cuda"])
        self.max_epochs = self.train_config['max_epochs']
        self.bs = train_config['batch_size']

        self.save_dir = f"{self.train_config['checkpoint_dir']}/{self.data_config['name']}_psrq.pt"
        # 确保保存目录存在
        Path(self.save_dir).parent.mkdir(parents=True, exist_ok=True)
        # 基本模型参数
        self.seq_len = self.data_config['seq_len']
        self.latent_dim = self.model_config['latent_dim']
        self.id_fields_num = self.data_config['id_fields_num']
        self.psrq_dims = self.model_config['psrq_dims']
        self.dropout = self.model_config['dropout']
        self.bn = self.model_config['batch_norm']
        self.projection_dim = self.model_config['projection_dim']
        self.id_feature_num = self.data_config['id_feature_num']

        # 模态参数，序列和非序列相同
        self.mm_features = self.data_config['use_mm_features']
        self.mm_nums = len(self.mm_features)
        self.mm_dims = self.data_config['mm_dims']

        # 新增参数
        self.n_levels = model_config.get('n_levels', 3)
        self.codebook_size = model_config.get('codebook_size', 1024)
        self.mu= model_config.get('mu', 0.25)
        self.quant_loss_weight = model_config.get('quant_loss_weight', 1.0)
        self.num_emb_list = [self.codebook_size for _ in range(self.n_levels)]
        self.mm_psrq = torch.nn.ModuleDict({k: PSRQ(
            in_dim=self.mm_dims[k],
            num_emb_list=self.num_emb_list,
            e_dim=self.projection_dim,
            layers=self.psrq_dims,
            dropout=self.dropout,
            bn=self.bn,
            loss_type="mse",
            quant_loss_weight=self.quant_loss_weight,
            mu = self.mu
        ) for k in self.mm_features if k != 'id'})

        self.joint_psrq = PSRQ(
            in_dim=sum([self.mm_dims[k] for k in self.mm_features if k != 'id']),
            num_emb_list=self.num_emb_list,
            e_dim=self.projection_dim,
            layers=self.psrq_dims,
            dropout=self.dropout,
            bn=True,
            loss_type="mse",
            quant_loss_weight=self.quant_loss_weight,
            mu = self.mu
        )

        self.compile()
        self.model_to_device()

    def save(self):
        torch.save(self.state_dict(), self.save_dir)

    def compile(self):
        self.optim = helper.getOptim(self, self.train_config['optim'],
                                     self.train_config["lr"], self.train_config["l2"])

    def load(self):
        self.load_state_dict(torch.load(self.save_dir, map_location=self.device, weights_only=True))

    def model_to_device(self):
        self.to(device=self.device)

    def set_up(self):
        helper.setup_seed(self.train_config['seed'])

    def fit(self, data_loader):
        best_auc = 0.0
        mm_modals = data_loader.get_multi_modal()
        mm_modals = {k: torch.from_numpy(mm_modals[k]).to(self.device) for k in self.mm_features if k != 'id'}
        print("text_shape: {}, image_shape: {}".format(mm_modals['text'].shape, mm_modals['image'].shape))
        print('mm_models_device:{}'.format(mm_modals['text'].device))
        total_samples = mm_modals[next(iter(mm_modals))].size(0)
        batch_size = self.bs
        for epoch_idx in range(self.max_epochs):
            print("Epoch: {}".format(epoch_idx))
            self.train()

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)

                # 构造当前 batch 的多模态输入
                batch_mm = {
                    k: v[start:end] for k, v in mm_modals.items()
                }

                self.optim.zero_grad()
                # 假设你的 forward 接受 batch_mm 字典
                losses = self(batch_mm)  # 返回 dict of losses
                loss = sum(losses.values())

                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()
                num_batches += 1
            avg_loss = epoch_loss / num_batches
            print(f"Average training loss: {avg_loss:.6f}")
        return best_auc

    def forward(self, mm_modals):
        mm_recons = {}
        mm_rq_loss = {}
        mm_indices = {}
        mm_x_q = {}
        for k in self.mm_features:
            if k != 'id':
                recon_mm_modal, mm_modal_rq_loss, mm_modal_indices, mm_modal_x_q = self.mm_psrq[k](mm_modals[k])
                mm_recons[k] = recon_mm_modal
                mm_rq_loss[k] = mm_modal_rq_loss
                mm_indices[k] = mm_modal_indices
                mm_x_q[k] = mm_modal_x_q

        mm_joint = torch.cat([mm_modals[k] for k in self.mm_features if k != 'id'], dim=1)

        recon_item_joint, joint_rq_loss, joint_indices, joint_x_q = self.joint_psrq(mm_joint)
        losses = {}
        for k in self.mm_features:
            if k != 'id':
                loss, _, _ = self.mm_psrq[k].compute_loss(mm_recons[k], mm_rq_loss[k], mm_modals[k])
                losses[k] = loss

        joint_loss, _, _ = self.joint_psrq.compute_loss(recon_item_joint, joint_rq_loss, mm_joint)
        losses['joint'] = joint_loss

        # print(torch.sum(self.text_psrq.psrq.vq_layers[0].embedding.weight.data))
        '''print(text_loss)
        print(text_loss.shape)
        print(joint_loss)'''
        '''for k in losses:
            losses[k] = losses[k].mean()'''

        return losses

    @torch.no_grad()
    def get_encode(self, feats, feats_seq):
        feats_indices = {}
        for m in self.mm_features:
            if m != 'id':
                feats_indices[m] = self.mm_psrq[m].get_indices(feats[m])

        mm_joint = torch.cat([feats[k] for k in self.mm_features if k != 'id'], dim=-1)
        joint_indices = self.joint_psrq.get_indices(mm_joint)

        # Handle feats_seq: (B, seq_num, dim)
        B, seq_num = feats_seq[self.mm_features[0]].shape[:2]  # assume all modalities share same B, seq_num
        feats_seq_indices = {}
        for m in self.mm_features:
            if m != 'id':
                # Flatten to (B * seq_num, dim_m)
                orig_shape = feats_seq[m].shape  # (B, seq_num, dim_m)
                flat_feats = feats_seq[m].view(-1, orig_shape[-1])
                flat_indices = self.mm_psrq[m].get_indices(flat_feats)
                # Reshape back to (B, seq_num)
                feats_seq_indices[m] = flat_indices.view(B, seq_num, -1)

        '''# Joint for feats_seq
        seq_feats_list = [feats_seq[k] for k in self.mm_features if k != 'id']  # each: (B, seq_num, dim_k)
        mm_joint_seq = torch.cat(seq_feats_list, dim=-1)  # (B, seq_num, sum_dim)
        flat_joint_seq = mm_joint_seq.view(-1, mm_joint_seq.size(-1))  # (B * seq_num, sum_dim)
        flat_joint_indices_seq = self.joint_psrq.get_indices(flat_joint_seq)
        joint_indices_seq = flat_joint_indices_seq.view(B, seq_num, -1)'''

        return feats_indices, joint_indices, feats_seq_indices


class PSRQ(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 mu=0.25,
                 ):
        super(PSRQ, self).__init__()

        self.in_dim = in_dim
        self.layers = layers
        self.dropout = dropout
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.e_dim = e_dim
        self.num_emb_list = num_emb_list
        self.encode_layer_dims = self.layers + [self.e_dim]
        self.encoder = MultiLayerPerceptron(self.in_dim, self.encode_layer_dims,
                                            dropout=self.dropout, use_bn=self.bn)

        self.psrq = PSResidualVectorQuantizer(self.num_emb_list, self.e_dim, mu=mu)

        self.decode_layer_dims = self.encode_layer_dims[::-1]  # 倒序
        self.decoder = MultiLayerPerceptron(self.encode_layer_dims[-1], self.decode_layer_dims[1:] + [self.in_dim],
                                            dropout=self.dropout, use_bn=self.bn)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x_q, rq_loss, indices = self.psrq(x)
        # print(x_q.shape, indices.shape)
        out = self.decoder(x_q)
        # print(out.shape)

        return out, rq_loss, indices, x_q

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, indices = self.psrq(x_e)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        rqvae_n_loss = loss_recon + self.quant_loss_weight * quant_loss

        total_loss = rqvae_n_loss

        return total_loss, loss_recon, quant_loss


class PSResidualVectorQuantizer(nn.Module):
    """
    Progressive Semantic Residual Quantizer (PSRQ)
    """

    def __init__(self, n_e_list, e_dim, mu=0.25, projector_hidden=None):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_levels = len(n_e_list)
        e_dim_list = [e_dim]
        for _ in range(self.num_levels - 1):
            e_dim_list.append(2 * e_dim)

        # VQ layers (all expect input e_dim)
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, mu=mu, kmeans_init=True) for n_e, e_dim in zip(n_e_list, e_dim_list)
        ])

        # Projectors for levels >=1 (2*e_dim -> e_dim)
        self.projectors = nn.ModuleList()
        for i in range(self.num_levels - 1):
            if projector_hidden is None:
                self.projectors.append(nn.Linear(2 * self.e_dim, self.e_dim))
            else:
                self.projectors.append(nn.Sequential(
                    nn.Linear(2 * self.e_dim, projector_hidden),
                    nn.ReLU(),
                    nn.Linear(projector_hidden, self.e_dim)
                ))
            # optional LayerNorm
            self.projectors[i] = nn.Sequential(self.projectors[i], nn.LayerNorm(self.e_dim))

    def get_codebook(self):
        all_codebook = [vq.get_codebook() for vq in self.vq_layers]
        return all_codebook

    def vq_ini(self, x):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            x_res = quantizer.vq_init(residual)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x):
        B = x.size(0)
        device = x.device

        residual = x  # (B, e_dim)
        prefix = torch.zeros_like(x)  # (B, e_dim)
        x_q = torch.zeros_like(x)  # accumulated quantized output

        all_losses = []
        all_indices = []

        for lvl in range(self.num_levels):
            if lvl == 0:
                vq_input = residual
                x_res, loss, indices = self.vq_layers[lvl](vq_input)
            else:
                vq_input = torch.cat([residual, prefix], dim=-1)  # (B, 2*e_dim)
                x_res, loss, indices = self.vq_layers[lvl](vq_input)
                x_res = self.projectors[lvl - 1](x_res)  # (B, e_dim)

            # 更新 residual / prefix / accumulation
            residual = residual - x_res
            # prefix = prefix + x_res
            prefix = x - residual
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices.view(B, -1) if indices.dim() == 1 else indices)

        # 合并 indices -> (B, num_levels)
        all_indices = torch.cat([idx.view(B, 1) if idx.dim() == 1 else idx.view(B, -1)
                                 for idx in all_indices], dim=1)
        mean_loss = torch.stack(all_losses).mean()

        return x_q, mean_loss, all_indices


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu=0.25, beta=1, kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.register_buffer('initted', torch.tensor(not kmeans_init, dtype=torch.bool))
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.register_buffer('initted', torch.tensor(not kmeans_init, dtype=torch.bool))
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        # centers = kmeans(
        #     data,
        #     self.n_e,
        #     self.kmeans_iters,
        # )
        centers, _ = self.constrained_km(data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted.fill_(True)

    def constrained_km(self, data, n_clusters=10):
        # from k_means_constrained import KMeansConstrained
        # x = data.cpu().detach().numpy()

        # size_min = min(len(data) // (n_clusters * 2), 50)  # 50 for the very first time, 10 the latter
        #
        # clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 4, max_iter=10, n_init=10,
        #                         n_jobs=10,
        #                         verbose=False)  # 'size_min * 4' for the very first time, 'n_clusters * 4' for the latter
        # clf.fit(x)
        # t_centers = torch.from_numpy(clf.cluster_centers_)
        # t_labels = torch.from_numpy(clf.labels_).tolist()
        from sklearn.cluster import KMeans
        x = data.cpu().detach().numpy()
        clf = KMeans(n_clusters=n_clusters, max_iter=10, n_init=10, verbose=0, random_state=2025)
        clf.fit(x)

        t_centers = torch.from_numpy(clf.cluster_centers_).to(data.dtype)  # 匹配输入的dtype
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels

    def vq_init(self, x):
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = 'distance'

        if _distance_flag == 'distance':
            d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()

        if _distance_flag == 'distance':
            indices = torch.argmin(d, dim=-1)
        else:
            indices = torch.argmax(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        return x_q

    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = 'distance'

        if _distance_flag == 'distance':
            d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()

        indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
