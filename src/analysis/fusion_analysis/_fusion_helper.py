"""
融合分析通用工具 —— 为不同融合方法提供统一的初始化和前向接口。

融合方法分类：
  - 本地融合 (LOCAL): 应用于单个物品的特征，dict/stack → tensor
    maf, cat, lmf, src, mtfn, fq-former, simcen

  - 序列感知融合 (SEQ): 需要序列上下文，替代模型的融合+池化步骤
    dta, gmmf, dmf
"""
import torch
import torch.nn as nn
from models.layers import modal_fusion

# 本地融合方法：应用于单个物品的特征融合
LOCAL_FUSIONS = {'maf', 'cat', 'lmf', 'src', 'mtfn', 'fq-former', 'simcen'}

# 序列感知融合方法：需要序列上下文
SEQ_FUSIONS = {'dta', 'gmmf', 'dmf'}

ALL_FUSIONS = LOCAL_FUSIONS | SEQ_FUSIONS


def normalize_fusion_method(method):
    """Normalize and validate fusion method names used by analysis adapters."""
    method = str(method).strip().lower().replace('_', '-')
    if method not in ALL_FUSIONS:
        raise ValueError(
            f"Unsupported fusion method '{method}'. Expected one of {sorted(ALL_FUSIONS)}")
    return method


def build_fusion(method, projection_dim, mm_features, model_config):
    """
    构建融合层（本地或序列感知）。

    返回 dict:
        'layer'      : nn.Module  融合层
        'out_dim'    : int        输出维度
        'query_num'  : int|None   FQ-Former 的 query 数
        'is_local'   : bool       是否为本地融合
        'sim_fusion' : nn.Module|None  DTA 所需的辅助融合层（计算多模态相似度）
        'seq_pooling': nn.Module|None  GMMF 所需的序列池化层
    """
    method = normalize_fusion_method(method)
    info = {
        'query_num': None,
        'sim_fusion': None,
        'seq_pooling': None,
        'is_local': method in LOCAL_FUSIONS,
    }

    if method == 'fq-former':
        info['query_num'] = model_config.get('query_num', 3)
        layer = modal_fusion.get_fusion_layer(
            'fq-former',
            dim=projection_dim,
            query_num=info['query_num'],
            layer_num=model_config.get('fq_layer_num', 2),
            num_heads=model_config.get('fq_num_heads', 8),
        )
        info['layer'] = layer
        info['out_dim'] = layer.getDim()

    elif method == 'simcen':
        layer = modal_fusion.get_fusion_layer(
            'simcen',
            num_fields=len(mm_features),
            embedding_dim=projection_dim,
            hidden_units=model_config.get('simcen_hidden_units', [480, 480, 480]),
            cl_temperature=model_config.get('simcen_cl_temp', 0.1),
        )
        info['layer'] = layer
        info['out_dim'] = layer.getDim()

    elif method == 'dta':
        layer = modal_fusion.get_fusion_layer(
            'dta',
            dim=projection_dim,
            mm_features=mm_features,
            attention_dim=model_config.get('attention_dim', 128),
            num_buckets=model_config.get('num_buckets', 35),
            dropout=model_config.get('dta_dropout', 0),
        )
        info['layer'] = layer
        info['out_dim'] = layer.getDim()
        # DTA 需要辅助融合层计算多模态相似度
        non_id = [k for k in mm_features if k != 'id']
        if non_id:
            info['sim_fusion'] = modal_fusion.get_fusion_layer('cat', projection_dim, non_id)

    elif method == 'gmmf':
        from models.layers import seq_pooling
        layer = modal_fusion.get_fusion_layer(
            'gmmf',
            dim=projection_dim,
            mm_features=mm_features,
            user_dim=model_config.get('gmmf_user_dim', None),
        )
        info['layer'] = layer
        info['out_dim'] = layer.getDim()
        info['seq_pooling'] = seq_pooling.get_pooling(
            'din', dim=projection_dim,
            mlp_dims=model_config.get('mlp_dims', [1024, 512, 256]),
        )

    elif method == 'dmf':
        layer = modal_fusion.get_fusion_layer(
            'dmf',
            dim=projection_dim,
            modal_fusion_method=model_config.get('dmf_inner_fusion', 'cat'),
            mm_features=mm_features,
            attention_dim=model_config.get('attention_dim', 128),
            num_buckets=model_config.get('num_buckets', 35),
            tier_num=model_config.get('tier_num', 10),
            mlp_dims=model_config.get('dmf_mlp_dims', [1024, 512, 256]),
            dropout=model_config.get('dropout', 0.0),
            alpha=model_config.get('alpha', 0.5),
        )
        info['layer'] = layer
        info['out_dim'] = layer.getDim()

    else:
        # Standard: maf, cat, lmf, src, mtfn
        kwargs = {}
        if method == 'lmf':
            kwargs['rank'] = model_config.get('rank', 5)
            kwargs['output_dim'] = model_config.get('fusion_dim', projection_dim)
        elif method == 'mtfn':
            kwargs['rank'] = model_config.get('rank', 20)
        elif method == 'src':
            kwargs['T'] = model_config.get('T', 10)
        layer = modal_fusion.get_fusion_layer(method, projection_dim, mm_features, **kwargs)
        info['layer'] = layer
        info['out_dim'] = layer.getDim()

    return info


def fuse_local(layer, feats_dict, mm_features, method, query_num=None):
    """
    应用本地融合到单个物品的特征。

    Args:
        layer: 融合层
        feats_dict: {modality: (B, D)} 特征字典
        mm_features: 模态名称列表
        method: 融合方法名
        query_num: FQ-Former query 数

    Returns:
        (fused_tensor, aux_loss)
        fused_tensor: (B, fusion_out_dim)
        aux_loss: 标量或 Tensor（simcen 有 cl_loss，其他为 0）
    """
    method = normalize_fusion_method(method)
    if method in SEQ_FUSIONS:
        raise ValueError(
            f"Fusion method '{method}' is sequence-aware and cannot be used by fuse_local(). "
            "Call the corresponding fuse_seq_* helper instead.")

    if method == 'fq-former':
        stacked = torch.stack([feats_dict[k] for k in mm_features], dim=1)  # (B, M, D)
        out = layer(stacked)  # (B, Q+M, D)
        fused = out[:, :query_num].reshape(stacked.size(0), -1)  # (B, Q*D)
        return fused, 0.0

    elif method == 'simcen':
        stacked = torch.stack([feats_dict[k] for k in mm_features], dim=1)  # (B, M, D)
        result = layer(stacked)
        return result['output'], result['cl_loss']

    else:
        # maf, cat, lmf, src, mtfn
        return layer(feats_dict), 0.0


def fuse_seq_dta(dta_layer, sim_fusion, feats_p, feats_seq_p, mm_features, seq_len):
    """
    DTA 序列感知融合。
    用非ID模态计算相似度，用ID嵌入做注意力池化。

    Returns:
        (user_interest, aux_loss)
        user_interest: (B, attention_dim)
    """
    non_id = [k for k in mm_features if k != 'id']

    # 计算目标物品和序列物品的多模态相似度
    target_mm = sim_fusion({k: feats_p[k] for k in non_id})  # (B, cat_dim)
    seq_mm = torch.stack(
        [sim_fusion({k: feats_seq_p[k][:, i] for k in non_id})
         for i in range(seq_len)],
        dim=1,
    )  # (B, L, cat_dim)
    similarity = torch.cosine_similarity(target_mm.unsqueeze(1), seq_mm, dim=2)  # (B, L)

    # DTA: 用 ID 嵌入 + 相似度分数
    user_interest = dta_layer(feats_p['id'], feats_seq_p['id'], similarity)
    return user_interest, 0.0


def fuse_seq_gmmf(gmmf_layer, gmmf_pooling, feats_p, feats_seq_p, user_vec):
    """
    GMMF 序列感知融合。

    Returns:
        (fused, aux_loss)
        fused: (B, gmmf_out_dim)
    """
    result = gmmf_layer(feats_p, feats_seq_p, user_vec, gmmf_pooling)
    # 收集辅助损失
    recon_loss = sum(
        torch.nn.functional.mse_loss(recon, target)
        for recon, target in result['recon_loss_pairs']
    )
    disc_loss = gmmf_layer.compute_disc_loss(
        feats_p, feats_seq_p,
        result['H_m'], result['H_m_hat'],
        result['H_m_seq'], result['H_m_seq_hat'],
    )
    gen_loss = gmmf_layer.compute_gen_loss(feats_p, feats_seq_p)
    aux_loss = recon_loss + 0.1 * disc_loss + 0.1 * gen_loss
    return result['fused'], aux_loss


def fuse_seq_dmf(dmf_layer, feats_p, feats_seq_p, seq_len):
    """
    DMF 序列感知融合。

    Returns:
        (user_interest, aux_loss)
    """
    user_interest = dmf_layer(feats_p, feats_seq_p, seq_len)
    return user_interest, 0.0
