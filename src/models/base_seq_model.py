import torch, numpy as np
import warnings
from pathlib import Path
from sklearn import metrics
from abc import ABC, abstractmethod
from mmctr.utils import helper
from .layers.common import FeatureEmbedding


class BaseSeqModel(torch.nn.Module, ABC):
    def __init__(self,
                 model_config,
                 train_config,
                 data_config,
                 logger):
        super().__init__()
        warnings.warn(
            "models.base_seq_model.BaseSeqModel is a legacy compatibility class; "
            "migrate to mmctr.models.BaseSeqModel",
            DeprecationWarning,
            stacklevel=2,
        )

        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.logger = logger

        self.set_up()
        self.seq_modeling = True  # 是否为序列建模模型
        # 基本模型参数
        self.seq_len = self.data_config['seq_len']
        self.latent_dim = self.model_config.get('latent_dim', 128)
        self.id_fields_num = self.data_config['id_fields_num']
        self.mlp_dims = self.model_config.get('mlp_dims', [1024, 512, 256])
        self.dropout = self.model_config.get('dropout', 0.5)
        self.bn = self.model_config.get('batch_norm', False)
        self.projection_dim = self.model_config.get('projection_dim', 128)
        self.id_feature_num = self.data_config['id_feature_num']
        self.seq_pooling_method = self.model_config.get('seq_pooling_method', 'mean')

        # 基本模态参数
        # 模态参数，序列和非序列相同
        self.mm_features = self.data_config['use_mm_features']
        self.mm_nums = len(self.mm_features)
        # 序列模型的 dataloader (get_data_seq) 中，物品和序列特征维度一致，
        # 均为 mm_seq_dims；mm_dims 是非序列 dataloader 的维度，在此不适用。
        self.mm_dims = dict(self.data_config.get('mm_seq_dims', self.data_config['mm_dims']))
        self.mm_dims["id"] = int(self.latent_dim)

        # 用户侧特征，不参与建模时输入预测层前拼接
        self.user_features = self.data_config['user_features']
        self.user_features_num = len(self.user_features)
        self.user_features_dim = self.data_config['user_features_dim']
        self.user_features_dim["id"] = int(self.latent_dim)

        self.modal_fusion_method = self.model_config.get('modal_fusion_method', 'add')

        # 基本训练参数
        self.ckpt_path = Path(
            self.train_config['checkpoint_dir']) / f"{self.model_config['model_name']}.pt"
        #print(self.ckpt_path)
        #self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.early_stop_patience = self.train_config["early_stop_patience"]
        self.device = helper.getDevice(self.train_config["cuda"])
        self.max_epochs = self.train_config['max_epochs']

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.log_interval = self.train_config["log_interval"]

        # 基本模块
        # 模态投影对齐，不区分是否序列
        self.mm_projector = torch.nn.ModuleDict({k: torch.nn.Linear(self.mm_dims[k], self.projection_dim)
                                                 for k in self.mm_features})
        # 用户侧模态投影对齐
        self.user_projector = torch.nn.ModuleDict({k: torch.nn.Linear(self.user_features_dim[k], self.projection_dim)
                                                   for k in self.user_features})
        self.embedding = FeatureEmbedding(self.id_feature_num + 1, self.latent_dim)

        # ── 模态鲁棒性实验支持 ────────────────────────────────────────
        # 默认关闭：正常训练 / 评测时行为与原模型完全一致。
        # 由鲁棒性实验通过 enable_modal_drop(True) 开启后，离散编码类模型
        # （如 QARM / MCCA）会识别出被整体置零的"缺失模态"样本，并在模态
        # 投影前将其表征同步置零，使模态缺失真正生效。
        self.modal_drop_enabled = False

    def enable_modal_drop(self, enabled: bool = True):
        """开启 / 关闭模态缺失适配（供模态鲁棒性实验调用）。"""
        self.modal_drop_enabled = bool(enabled)

    def build_modal_keep_mask(self, feats: dict) -> dict:
        """构造每个非 ID 模态的样本级保留掩码 (1=保留, 0=缺失)。

        判定依据：物品级模态特征是否被整体置零（鲁棒性实验对缺失样本的
        feats[m] 与 feats_seq[m] 会一并置零，因此用物品级即可识别缺失样本）。
        仅供离散编码类模型在投影前同步置零缺失模态使用。
        """
        keep = {}
        modal_keys = getattr(self, 'rq_features',
                             [k for k in self.mm_features if k != 'id'])
        for m in modal_keys:
            keep[m] = (feats[m].abs().sum(dim=-1) != 0).to(feats[m].dtype)
        return keep

    def set_up(self):
        helper.setup_seed(self.train_config['seed'])

    def compile(self):
        self.optim = helper.getOptim(self, self.train_config['optim'],
                                     self.train_config["lr"], self.train_config["l2"])

    def model_to_device(self):
        self.to(device=self.device)

    def save(self):
        torch.save(self.state_dict(), self.ckpt_path)

    def load(self):
        self.load_state_dict(torch.load(self.ckpt_path, map_location=self.device, weights_only=True))

    @abstractmethod
    def forward(self, user_feats, feats, feats_seq) -> torch.Tensor:
        ...

    def fit(self, data_loader):
        patience = self.early_stop_patience
        best_auc = 0.0

        for epoch_idx in range(self.max_epochs):
            train_times = []
            self.train()
            train_loss = .0
            step = 0

            for batch in data_loader.get_data_seq('train'):
                with helper.timer(train_times):
                    self.optim.zero_grad()
                    out, y = self._predict_batch(batch)
                    au_loss = out.get('au_loss', 0)
                    loss = self.compute_loss(out['pred'], y) + au_loss
                    loss.backward()
                    self.optim.step()

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

    def compute_loss(self, logits, y) -> torch.Tensor:
        return self.criterion(logits, y.float())

    def _predict_batch(self, batch):
        user_feats, feats, feats_seq, label = batch
        user_feats = {k: v.to(self.device) for k, v in user_feats.items()}
        feats = {k: v.to(self.device) for k, v in feats.items()}
        feats_seq = {k: v.to(self.device) for k, v in feats_seq.items()}
        label = label.to(self.device)
        out = self(user_feats,feats, feats_seq )
        return out, label

    @torch.no_grad()
    def evalate(self, data_loader, on):
        self.eval()
        preds, trues = [], []
        inference_times = []  # 添加推理时间统计
        for batchs in data_loader.get_data_seq(on):
            with helper.timer(inference_times):
                out, label = self._predict_batch(batchs)
            pred = out['pred'].sigmoid().detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)

        # 计算平均推理时间
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            self.logger.info(f"平均Batch推理时间: {avg_inference_time * 1000:.2f} ms "
                             f"(共{len(inference_times)}个batch)")

        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def log_model_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fixed = total - train

        lines = [
            "=" * 60,
            "模型参数统计:",
            f"  总参数数量 : {total:,}",
            f"  可训练参数 : {train:,}",
            f"  非训练参数 : {fixed:,}",
            "=" * 60,
        ]
        out = "\n".join(lines)

        self.logger.info("\n" + out)
