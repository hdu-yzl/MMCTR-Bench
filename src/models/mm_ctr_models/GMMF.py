import numpy as np

from ..base_seq_model import BaseSeqModel
from ..layers.common import MultiLayerPerceptron
from ..layers import modal_fusion, seq_pooling
import torch.nn as nn
import torch
from utils import helper


class GMMF(BaseSeqModel):
    def __init__(self, model_config, train_config, data_config, logger):
        super().__init__(model_config, train_config, data_config, logger)
        self.lambdas = model_config.get('lambdas', {'image': 0.05, 'text': 0.05, 'audio': 0.05})
        self.N = model_config.get('N', 1)  # 更新GCN起始轮数
        self.lr_main = float(model_config.get('lr_main', 0.001))
        self.lr_disc = float(model_config.get('lr_disc', 0.0001))
        self.lr_gen = float(model_config.get('lr_gen', 0.0001))
        self.l2 = float(model_config.get('l2', 1e-7))

        self.modal_poolings = seq_pooling.get_pooling('cos')

        # DSN: 自编码器
        self.mm_ae = nn.ModuleDict({
            k: AutoEncoder(self.projection_dim, self.projection_dim)
            for k in self.mm_features if k != 'id'
        })

        # DSN: CGAN生成器
        self.mm_gen = nn.ModuleDict({
            k: CGANGenerator(self.projection_dim, self.projection_dim)
            for k in self.mm_features if k != 'id'
        })

        # DSN: CGAN判别器
        self.mm_disc = nn.ModuleDict({
            k: CGANDiscriminator(self.projection_dim)
            for k in self.mm_features if k != 'id'
        })

        # DSN: 自动差分模块
        self.mm_diff = nn.ModuleDict({
            k: AutoDifference(self.projection_dim)
            for k in self.mm_features if k != 'id'
        })

        # MIN: 门控网络
        user_feat_dim = self.projection_dim * self.user_features_num
        self.mm_gate = nn.ModuleDict({
            k: ModalInterestGate(user_feat_dim, self.projection_dim)
            for k in self.mm_features
        })

        final_input_dim = self.projection_dim * self.mm_nums + user_feat_dim

        self.dnn = MultiLayerPerceptron(final_input_dim, self.mlp_dims, self.dropout, use_bn=self.bn)
        self.out_put = MultiLayerPerceptron(self.mlp_dims[-1],
                                            [1], self.dropout,
                                            use_bn=self.bn, activation=None)
        self.relu = nn.ReLU()
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()  # 后面已经有sigmoid

        self.compile()  # 初始化优化器
        self.model_to_device()
        self.log_model_params()

    def forward(self, user_feats, feats, feats_seq):
        au_loss = 0.0
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # DSN: 自编码器编码
        H_m, H_m_seq = {}, {}
        I_m_recon, I_m_seq_recon = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m[k], I_m_recon[k] = self.mm_ae[k](feats_p[k])
            H_m_seq[k], I_m_seq_recon[k] = self.mm_ae[k](feats_seq_p[k])

        # DSN: CGAN生成跨模态特征
        H_m_hat, H_m_seq_hat = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m_hat[k] = self.mm_gen[k](feats_p['id'])
            H_m_seq_hat[k] = self.mm_gen[k](feats_seq_p['id'])

        # DSN: 自动差分去除冗余
        H_m_prime, H_m_seq_prime = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m_prime[k] = self.mm_diff[k](H_m[k], H_m_hat[k])
            H_m_seq_prime[k] = self.mm_diff[k](H_m_seq[k], H_m_seq_hat[k])

        # 用户序列进行attention pooling
        H_m_seq_pool = {k: self.modal_poolings(H_m_prime[k], H_m_seq_prime[k]) for k in self.mm_features if k != 'id'}

        # 用户侧
        user_feats['id'] = self.embedding(user_feats['id']).squeeze()
        user_feats = {k: self.user_projector[k](user_feats[k]) for k in self.user_features}
        user_tensors = [user_feats[k] for k in user_feats]
        user_vec = torch.cat(user_tensors, dim=1)

        # 门控
        mm_gate = {k: self.mm_gate[k](user_vec) for k in self.mm_features}

        tensors = []
        for k in self.mm_features:
            if k == 'id':
                tensors.append(feats_p[k] * mm_gate[k])
                continue
            tensors.append(H_m_seq_pool[k] * mm_gate[k])

        final_vec = torch.cat(tensors, dim=-1)
        final_vec = torch.cat((final_vec, user_vec), dim=-1)
        dnn_out = self.dnn(final_vec)
        logit = self.out_put(dnn_out)

        for k in self.mm_features:
            if k == 'id':
                continue
            au_loss += self.lambdas[k] * self.mse(I_m_recon[k], feats_p[k]) + self.lambdas[k] * self.mse(
                I_m_seq_recon[k],
                feats_seq_p[k])
        out = {'pred': logit, 'au_loss': au_loss}
        return out

    def compile(self):
        main_params = [p for n, p in self.named_parameters()
                       if 'disc.' not in n and 'gen.' not in n]
        # 判别器参数
        disc_params = [p for n, p in self.named_parameters() if 'disc.' in n]

        # 生成器参数
        gen_params = [p for n, p in self.named_parameters() if 'gen.' in n]

        self.optimizer_main = torch.optim.Adam(main_params, lr=float(self.train_config["lr"]), weight_decay=self.l2)
        self.optimizer_disc = torch.optim.Adam(disc_params, lr=self.lr_disc, weight_decay=self.l2)
        self.optimizer_gen = torch.optim.Adam(gen_params, lr=self.lr_gen, weight_decay=self.l2)

    # 判别器损失
    def comput_GAN_loss1(self, user_feats, feats, feats_seq):
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}

        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # DSN: 自编码器编码
        H_m, H_m_seq = {}, {}
        I_m_recon, I_m_seq_recon = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m[k], I_m_recon[k] = self.mm_ae[k](feats_p[k])
            H_m_seq[k], I_m_seq_recon[k] = self.mm_ae[k](feats_seq_p[k])

        # DSN: CGAN生成跨模态特征
        H_m_hat, H_m_seq_hat = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m_hat[k] = self.mm_gen[k](feats_p['id'])
            H_m_seq_hat[k] = self.mm_gen[k](feats_seq_p['id'])

        m_true, m_seq_true = {}, {}
        m_fake, m_seq_fake = {}, {}
        for i in self.mm_features:
            if i == 'id':
                continue
            m_true[i] = self.mm_disc[i](feats_p['id'], H_m[i])
            m_seq_true[i] = self.mm_disc[i](feats_seq_p['id'], H_m_seq[i])
            m_fake[i] = self.mm_disc[i](feats_p['id'], H_m_hat[i])
            m_seq_fake[i] = self.mm_disc[i](feats_seq_p['id'], H_m_seq_hat[i])

        GAN_loss = 0.0
        for i in self.mm_features:
            if i == 'id':
                continue
            GAN_loss += self.bce(m_true[i], torch.ones_like(m_true[i])) + self.bce(m_fake[i],
                                                                                   torch.zeros_like(m_fake[i])) + \
                        self.bce(m_seq_true[i], torch.ones_like(m_seq_true[i])) + self.bce(m_seq_fake[i],
                                                                                           torch.zeros_like(
                                                                                               m_seq_fake[i]))
        return GAN_loss

    # 生成器损失
    def comput_GAN_loss2(self, user_feats, feats, feats_seq):
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}
        feats['id'] = self.embedding(feats['id']).squeeze()  # (B,latent_dim) # itemid
        feats_seq['id'] = self.embedding(feats_seq['id'])  # (B,seq_num,latent_dim)

        feats_p = {k: self.mm_projector[k](feats[k]) for k in self.mm_features}
        feats_seq_p = {k: self.mm_projector[k](feats_seq[k]) for k in self.mm_features}

        # DSN: CGAN生成跨模态特征
        H_m_hat, H_m_seq_hat = {}, {}
        for k in self.mm_features:
            if k == 'id':
                continue
            H_m_hat[k] = self.mm_gen[k](feats_p['id'])
            H_m_seq_hat[k] = self.mm_gen[k](feats_seq_p['id'])

        m_fake, m_fake_seq = {}, {}
        for i in self.mm_features:
            if i == 'id':
                continue
            m_fake[i] = self.mm_disc[i](feats_p['id'], H_m_hat[i])
            m_fake_seq[i] = self.mm_disc[i](feats_seq_p['id'], H_m_seq_hat[i])

        GAN_loss = 0.0
        for i in self.mm_features:
            if i == 'id':
                continue
            GAN_loss += self.bce(m_fake[i], torch.ones_like(m_fake[i])) + self.bce(m_fake_seq[i],
                                                                                   torch.ones_like(m_fake_seq[i]))
        return GAN_loss

    def fit(self, data_loader):
        patience = self.early_stop_patience
        best_auc = 0.0
        train_times = []
        for epoch_idx in range(self.max_epochs):
            self.train()
            train_loss = .0
            step = 0

            for batch in data_loader.get_data_seq('train'):
                with helper.timer(train_times):
                    self.optimizer_main.zero_grad()
                    out, y = self._predict_batch(batch)
                    au_loss = out.get('au_loss', 0)
                    loss = self.compute_loss(out['pred'], y) + au_loss
                    loss.backward()
                    self.optimizer_main.step()

                    if epoch_idx >= self.N:
                        self.optimizer_disc.zero_grad()
                        GAN_loss = self.comput_GAN_loss1(batch[0], batch[1], batch[2])
                        GAN_loss.backward()
                        self.optimizer_disc.step()
                        train_loss += GAN_loss.item()

                        self.optimizer_gen.zero_grad()
                        GAN_loss = self.comput_GAN_loss2(batch[0], batch[1], batch[2])
                        GAN_loss.backward()
                        self.optimizer_gen.step()
                        train_loss += GAN_loss.item()

                    train_loss += loss.item()
                    step += 1
                    if step % self.log_interval == 0:
                        self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                         format(epoch=epoch_idx, setp=step, loss=loss))
            if train_times:
                avg_train_time = np.mean(train_times)
                self.logger.info(f"平均Batch训练时间: {avg_train_time * 1000:.2f} ms "
                                 f"(共{len(train_times)}个batch)")
            train_loss /= step
            val_auc, val_loss = self.evalate(data_loader, "val")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

            if val_auc > best_auc:
                best_auc, patience = val_auc, self.early_stop_patience
                self.save()
            else:
                patience -= 1
                if patience <= 0:
                    self.logger.info(f'Early stop at epoch {epoch_idx}')
                    break
        self.load()
        return best_auc


class AutoEncoder(nn.Module):
    """自编码器模块，用于降维和特征重建"""

    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()  # 必须加

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        encoded_relu = self.relu(encoded)
        decoded_relu = self.relu(decoded)
        return encoded_relu, decoded_relu


class CGANGenerator(nn.Module):
    """CGAN生成器，将ID嵌入映射到多模态空间"""

    def __init__(self, id_dim, output_dim):
        super(CGANGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(id_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, id_embedding):
        return self.net(id_embedding)


class CGANDiscriminator(nn.Module):
    """CGAN判别器，判断(ID, 模态)对的真实性"""

    def __init__(self, modality_dim):
        super(CGANDiscriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(modality_dim * 4, modality_dim * 2),
            nn.ReLU(),
            nn.Linear(modality_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, id_, modality):
        x = torch.cat([id_, id_ - modality, id_ * modality, modality], dim=-1)
        return self.net(x)


class AutoDifference(nn.Module):
    """自动差分模块，学习去除冗余信息"""

    def __init__(self, modality_dim):
        super(AutoDifference, self).__init__()
        self.weight_net = nn.Linear(modality_dim * 4, modality_dim)

    def forward(self, Hm, Hm_hat):
        # 拼接原始特征、生成特征、相似度和差异度
        concat_feat = torch.cat([
            Hm,
            Hm_hat,
            Hm * Hm_hat,  # 元素乘积，衡量相似性
            Hm - Hm_hat  # 差异度
        ], dim=-1)
        weights = torch.sigmoid(self.weight_net(concat_feat))
        return Hm - weights * Hm_hat


class ModalInterestGate(nn.Module):
    """MIN门控网络，基于用户嵌入生成模态偏好权重"""

    def __init__(self, user_dim, modal_dim):
        super(ModalInterestGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(user_dim, modal_dim),
            nn.ReLU(),
            nn.Linear(modal_dim, modal_dim)
        )

    def forward(self, user_embedding):
        return torch.softmax(self.net(user_embedding), dim=-1)
