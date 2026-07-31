import torch.nn as nn
import torch.nn.functional as F
import torch
class HingeCosineLoss(nn.Module):
    def __init__(self):
        super(HingeCosineLoss, self).__init__()

    def forward(self, tensor1, tensor2, labels):
        labels = labels.view(-1)  # 确保 (B,)，避免 (B,1) 与 (B,) 广播为 (B,B)
        cos_sim = F.cosine_similarity(tensor1, tensor2, dim=-1)  # (B,)
        pos = F.relu(1 - cos_sim)      # 正样本：推动 cos_sim → 1
        neg = F.relu(cos_sim + 1)      # 负样本：推动 cos_sim → -1
        loss = (1 - labels) * neg + labels * pos
        return loss.mean()

class CIC_Loss(nn.Module):
    def __init__(self, cic_tau):
        super(CIC_Loss, self).__init__()
        self.cic_tau = cic_tau

    def forward(self, tensor1, tensor2):
        bs = tensor1.size(0)
        score_c2i = torch.matmul(tensor1, tensor2.t()) / self.cic_tau  # (B,B)
        score_i2c = score_c2i.t()
        label = torch.arange(bs, device=tensor1.device)  # (B)

        cic_loss = (torch.nn.functional.cross_entropy(score_c2i, label) +
                    torch.nn.functional.cross_entropy(score_i2c, label)) / 2
        return cic_loss