import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceAlignment(nn.Module):
    """
    KL散度对齐损失
    将所有非ID模态与ID模态对齐，使其分布接近ID模态分布
    """
    def __init__(self, mm_features=['id', 'text', 'image'], temperature=1.0, lambda_weight=0.1):
        """
        Args:
            mm_features: 模态名称列表，必须包含'id'
            temperature: 温度参数，用于平滑分布
            lambda_weight: 损失权重系数
        """
        super().__init__()
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']
        self.temperature = temperature
        self.lambda_weight = lambda_weight
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, mm_feats: dict):
        """
        Args:
            mm_feats: 模态特征字典 {modality: tensor[B, D]}
        Returns:
            加权后的KL散度对齐损失
        """
        if 'id' not in mm_feats:
            raise ValueError("mm_feats must contain 'id' modality")
        
        id_emb = mm_feats['id']
        total_loss = 0.0
        
        # 将ID模态转为概率分布（作为目标分布）
        id_prob = F.softmax(id_emb / self.temperature, dim=-1)
        
        # 计算每个非ID模态与ID模态的KL散度
        for modality in self.non_id:
            if modality in mm_feats:
                modal_log_prob = F.log_softmax(mm_feats[modality] / self.temperature, dim=-1)
                total_loss += self.kl_loss(modal_log_prob, id_prob)
        
        # 平均并加权
        if len(self.non_id) > 0:
            total_loss = total_loss / len(self.non_id)
        
        return self.lambda_weight * total_loss


class ContrastiveAlignment(nn.Module):
    """
    对比学习对齐
    通过InfoNCE损失将所有非ID模态与ID模态对齐
    """
    def __init__(self, mm_features=['id', 'text', 'image'], temperature=0.07, lambda_weight=0.1):
        """
        Args:
            mm_features: 模态名称列表，必须包含'id'
            temperature: 对比学习温度参数
            lambda_weight: 损失权重系数
        """
        super().__init__()
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']
        self.temperature = temperature
        self.lambda_weight = lambda_weight
    
    def forward(self, mm_feats: dict):
        """
        Args:
            mm_feats: 模态特征字典 {modality: tensor[B, D]}
        Returns:
            加权后的对比学习对齐损失
        """
        if 'id' not in mm_feats:
            raise ValueError("mm_feats must contain 'id' modality")
        
        id_emb = F.normalize(mm_feats['id'], dim=-1)
        batch_size = id_emb.size(0)
        labels = torch.arange(batch_size, device=id_emb.device)
        
        total_loss = 0.0
        
        # 计算每个非ID模态与ID模态的对比损失
        for modality in self.non_id:
            if modality in mm_feats:
                modal_emb = F.normalize(mm_feats[modality], dim=-1)
                
                # 计算相似度矩阵
                logits = torch.matmul(id_emb, modal_emb.t()) / self.temperature  # (B, B)
                
                # 双向对比损失
                loss_id2modal = F.cross_entropy(logits, labels)
                loss_modal2id = F.cross_entropy(logits.t(), labels)
                
                total_loss += (loss_id2modal + loss_modal2id) / 2
        
        # 平均并加权
        if len(self.non_id) > 0:
            total_loss = total_loss / len(self.non_id)
        
        return self.lambda_weight * total_loss


class CosineAlignment(nn.Module):
    """
    余弦相似度对齐
    最大化所有非ID模态与ID模态之间的余弦相似度
    """
    def __init__(self, mm_features=['id', 'text', 'image'], margin=0.0, lambda_weight=0.1):
        """
        Args:
            mm_features: 模态名称列表，必须包含'id'
            margin: 边界值，用于控制相似度的最小阈值
            lambda_weight: 损失权重系数
        """
        super().__init__()
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']
        self.margin = margin
        self.lambda_weight = lambda_weight
    
    def forward(self, mm_feats: dict):
        """
        Args:
            mm_feats: 模态特征字典 {modality: tensor[B, D]}
        Returns:
            加权后的余弦相似度对齐损失
        """
        if 'id' not in mm_feats:
            raise ValueError("mm_feats must contain 'id' modality")
        
        id_emb = mm_feats['id']
        total_loss = 0.0
        
        # 计算每个非ID模态与ID模态的余弦相似度损失
        for modality in self.non_id:
            if modality in mm_feats:
                cos_sim = F.cosine_similarity(id_emb, mm_feats[modality], dim=-1)
                loss = torch.clamp(1.0 - cos_sim + self.margin, min=0.0)
                total_loss += loss.mean()
        
        # 平均并加权
        if len(self.non_id) > 0:
            total_loss = total_loss / len(self.non_id)
        
        return self.lambda_weight * total_loss


class MMDAlignment(nn.Module):
    """
    最大均值差异(Maximum Mean Discrepancy)对齐
    通过核方法度量所有非ID模态与ID模态分布之间的距离
    """
    def __init__(self, mm_features=['id', 'text', 'image'], 
                 kernel_type='rbf', kernel_mul=2.0, kernel_num=5, lambda_weight=0.1):
        """
        Args:
            mm_features: 模态名称列表，必须包含'id'
            kernel_type: 核函数类型，支持'rbf'和'linear'
            kernel_mul: 多核的倍数
            kernel_num: 核的数量
            lambda_weight: 损失权重系数
        """
        super().__init__()
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.lambda_weight = lambda_weight
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算高斯核矩阵
        """
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        
        # 计算样本间的L2距离
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # 计算多个带宽
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        # 计算多核矩阵
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def linear_kernel(self, source, target):
        """
        计算线性核矩阵
        """
        total = torch.cat([source, target], dim=0)
        kernel_val = torch.matmul(total, total.t())
        return kernel_val
    
    def forward(self, mm_feats: dict):
        """
        Args:
            mm_feats: 模态特征字典 {modality: tensor[B, D]}
        Returns:
            加权后的MMD对齐损失
        """
        if 'id' not in mm_feats:
            raise ValueError("mm_feats must contain 'id' modality")
        
        id_emb = mm_feats['id']
        batch_size = int(id_emb.size(0))
        total_loss = 0.0
        
        # 计算每个非ID模态与ID模态的MMD
        for modality in self.non_id:
            if modality in mm_feats:
                modal_emb = mm_feats[modality]
                
                # 选择核函数
                if self.kernel_type == 'rbf':
                    kernels = self.gaussian_kernel(id_emb, modal_emb, 
                                                  kernel_mul=self.kernel_mul, 
                                                  kernel_num=self.kernel_num)
                elif self.kernel_type == 'linear':
                    kernels = self.linear_kernel(id_emb, modal_emb)
                else:
                    raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
                
                # 分割核矩阵
                XX = kernels[:batch_size, :batch_size]
                YY = kernels[batch_size:, batch_size:]
                XY = kernels[:batch_size, batch_size:]
                YX = kernels[batch_size:, :batch_size]
                
                # 计算MMD
                mmd = XX.mean() + YY.mean() - XY.mean() - YX.mean()
                total_loss += mmd
        
        # 平均并加权
        if len(self.non_id) > 0:
            total_loss = total_loss / len(self.non_id)
        
        return self.lambda_weight * total_loss


class AdversarialAlignment(nn.Module):
    """
    对抗对齐
    通过判别器对抗训练将所有非ID模态与ID模态对齐
    """
    def __init__(self, input_dim, mm_features=['id', 'text', 'image'], 
                 hidden_dims=None, dropout=0.1, lambda_weight=0.1):
        """
        Args:
            input_dim: 输入维度
            mm_features: 模态名称列表，必须包含'id'
            hidden_dims: 判别器隐藏层维度列表
            dropout: dropout比例
            lambda_weight: 损失权重系数
        """
        super().__init__()
        self.mm_features = mm_features
        self.non_id = [k for k in mm_features if k != 'id']
        self.lambda_weight = lambda_weight
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 为每个非ID模态构建独立的判别器
        self.discriminators = nn.ModuleDict()
        for modality in self.non_id:
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.discriminators[modality] = nn.Sequential(*layers)
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, mm_feats: dict, mode='generator'):
        """
        Args:
            mm_feats: 模态特征字典 {modality: tensor[B, D]}
            mode: 'discriminator' 或 'generator'
                - discriminator: 训练判别器，区分ID模态和其他模态
                - generator: 训练特征提取器，混淆判别器
        Returns:
            加权后的对抗对齐损失
        """
        if 'id' not in mm_feats:
            raise ValueError("mm_feats must contain 'id' modality")
        
        id_emb = mm_feats['id']
        batch_size = id_emb.size(0)
        total_loss = 0.0
        
        for modality in self.non_id:
            if modality in mm_feats:
                modal_emb = mm_feats[modality]
                
                # 判别器预测
                id_pred = self.discriminators[modality](id_emb)
                modal_pred = self.discriminators[modality](modal_emb)
                
                if mode == 'discriminator':
                    # 判别器损失：正确区分ID模态和其他模态
                    id_labels = torch.ones(batch_size, 1, device=id_emb.device)
                    modal_labels = torch.zeros(batch_size, 1, device=modal_emb.device)
                    
                    loss_id = self.bce_loss(id_pred, id_labels)
                    loss_modal = self.bce_loss(modal_pred, modal_labels)
                    
                    total_loss += (loss_id + loss_modal) / 2
                
                elif mode == 'generator':
                    # 生成器损失：混淆判别器（让其他模态被判别为ID模态）
                    modal_labels = torch.ones(batch_size, 1, device=modal_emb.device)
                    total_loss += self.bce_loss(modal_pred, modal_labels)
                
                else:
                    raise ValueError(f"Unsupported mode: {mode}. Use 'discriminator' or 'generator'.")
        
        # 平均并加权
        if len(self.non_id) > 0:
            total_loss = total_loss / len(self.non_id)
        
        return self.lambda_weight * total_loss
    
    def get_discriminator_loss(self, mm_feats: dict):
        """
        获取判别器损失（用于训练判别器）
        """
        return self.forward(mm_feats, mode='discriminator')
    
    def get_generator_loss(self, mm_feats: dict):
        """
        获取生成器损失（用于训练特征提取器）
        """
        return self.forward(mm_feats, mode='generator')
