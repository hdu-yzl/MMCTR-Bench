import numpy as np
from sklearn.cluster import KMeans as SKKMeans
import torch
from pathlib import Path
class ResidualQuantizer:
    def __init__(self, model_config,
                 train_config,
                 data_config):
        """
        Args:
            n_levels: 残差量化层数L
            codebook_size: 每层codebook的大小N
            dimension: 向量维度F
            random_state: 随机种子
        """
        self.train_config = train_config
        self.n_levels = model_config.get('n_levels', 3)
        self.codebook_size = model_config.get('codebook_size', 1024)
        self.dimension = model_config.get('dimension', 128)
        self.random_state = model_config.get('random_state', 42)
        self.n_init = model_config.get('n_init', 3)
        self.max_iter = model_config.get('max_iter', 20)
        self.tol = float(model_config.get('tol', 1e-4))

        self.codebooks = []  # 存储每层codebook: [L, N, F]

    def _find_nearest_codes(self, vectors, codebook):
        """
        向量化暴力最近邻搜索
        计算所有样本到所有中心的距离，返回最近邻索引

        Args:
            vectors: shape (n_samples, dimension)
            codebook: shape (n_clusters, dimension)

        Returns:
            indices: shape (n_samples,) 最近邻中心索引
        """
        # 向量化计算欧氏距离: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c^T
        # 时间复杂度: O(n_samples * n_clusters * dimension) 但高度向量化
        x_squared = np.sum(vectors ** 2, axis=1, keepdims=True)  # (n_samples, 1)
        c_squared = np.sum(codebook ** 2, axis=1, keepdims=True)  # (n_clusters, 1)
        distances = x_squared + c_squared.T - 2 * np.dot(vectors, codebook.T)
        return np.argmin(distances, axis=1)

    def fit(self, data, verbose=True):
        """
        Args:
            data: numpy array, shape (n_samples, dimension)
        """
        if verbose:
            print(f"开始训练RQ量化器: {self.n_levels}层, codebook大小={self.codebook_size}")

        current_data = data.copy().astype(np.float32)

        for level in range(self.n_levels):
            if verbose:
                print(f"\n  训练第 {level + 1}/{self.n_levels} 层...")

            kmeans = SKKMeans(
                n_clusters=self.codebook_size,
                init='k-means++',
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state + level,
                verbose=0
            )

            # 训练K-means
            kmeans.fit(current_data)

            # 为每个样本查找最近的codebook条目索引
            indices = kmeans.predict(current_data)

            # 获取每个样本对应的中心点向量
            O = kmeans.cluster_centers_[indices]

            # 计算残差: M = M - O
            current_data = current_data - O

            self.codebooks.append(kmeans.cluster_centers_.copy())

            if verbose:
                residual_norm = np.linalg.norm(current_data) / np.sqrt(len(current_data))
                print(f"    平均残差: {residual_norm:.4f}")

        if verbose:
            print("\nRQ codebooks训练完成!")

    def encode(self, vectors):
        """
        Args:
            vectors: numpy array, shape (n_vectors, dimension)
                    要编码的向量m

        Returns:
            codes: numpy array, shape (n_vectors, n_levels)
                   每层的编码 [r1, r2, ..., rL]
            quantized: numpy array, shape (n_vectors, dimension)
                      重建的量化向量
        """
        if not self.codebooks:
            raise ValueError("Codebooks未训练! 请先调用train()方法。")

        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.n_levels), dtype=np.int64)
        residuals = vectors.copy().astype(np.float32)
        quantized_sum = np.zeros_like(vectors, dtype=np.float32)

        # 逐层量化
        for level in range(self.n_levels):
            codebook = self.codebooks[level]

            # 使用向量化暴力最近邻查找
            level_codes = self._find_nearest_codes(residuals, codebook)

            # 保存当前层编码: [r1, r2, ..., rL]
            codes[:, level] = level_codes

            # 获取对应中心点: R_i[r_i]
            level_centroids = codebook[level_codes]

            # 更新残差: m_i = m_{i-1} - R_i[r_i]
            residuals -= level_centroids

            # 累加量化结果
            quantized_sum += level_centroids

        return codes, quantized_sum

    def encode_tensor(self, vectors: torch.Tensor):
        """
        Args:
            vectors: torch.Tensor, shape (n_vectors, dimension)
                    要编码的向量 m

        Returns:
            codes: torch.LongTensor, shape (n_vectors, n_levels)
                   每层的编码 [r1, r2, ..., rL]
            quantized: torch.Tensor, shape (n_vectors, dimension)
                      重建的量化向量
        """
        if not self.codebooks:
            raise ValueError("Codebooks未训练! 请先调用train()方法。")

        device = vectors.device
        dtype = vectors.dtype

        # 一次性把 NumPy -> Tensor 并搬设备 (使用局部变量，不修改self.codebooks)
        codebooks_tensor = []
        for c in self.codebooks:
            if isinstance(c, np.ndarray):
                codebooks_tensor.append(torch.from_numpy(c).to(device, dtype))
            else:
                codebooks_tensor.append(c.to(device, dtype) if c.device != device else c)

        # 保存原始形状，便于最后恢复
        original_shape = vectors.shape
        # 将输入展平为 (N, dim)，其中 N = B 或 B * seq_num
        if vectors.dim() == 2:
            N, dim = vectors.shape
            flat_vectors = vectors
        elif vectors.dim() == 3:
            B, seq_num, dim = vectors.shape
            flat_vectors = vectors.view(-1, dim)  # (B * seq_num, dim)
            N = flat_vectors.shape[0]
        else:
            raise ValueError("输入张量维度必须为2或3")

        residuals = flat_vectors.clone()
        quantized_sum = torch.zeros_like(flat_vectors, dtype=dtype, device=device)
        codes = torch.zeros(N, self.n_levels, dtype=torch.long, device=device)

        for level in range(self.n_levels):
            codebook = codebooks_tensor[level]  # shape: (codebook_size, dim)

            # 计算残差与码本中每个向量的欧氏距离
            # residuals: (N, dim), codebook: (K, dim)
            # 利用广播计算 (N, K)
            diff = residuals.unsqueeze(1) - codebook.unsqueeze(0)  # (N, K, dim)
            distances = torch.norm(diff, dim=2)  # (N, K)

            # 找到最近的码字索引
            level_codes = torch.argmin(distances, dim=1)  # (N,)
            codes[:, level] = level_codes

            # 获取对应的码字
            level_centroids = codebook[level_codes]  # (N, dim)

            # 更新残差和量化结果
            residuals = residuals - level_centroids
            quantized_sum = quantized_sum + level_centroids

        # 恢复形状
        codes = codes.view(*original_shape[:-1], self.n_levels)  # (..., n_levels)
        quantized = quantized_sum.view(*original_shape)  # same as input

        return codes, quantized

    def decode(self, codes):
        """
        根据编码重建量化向量
        Args:
            codes: numpy array, shape (n_vectors, n_levels)
                   RQ编码 [r1, r2, ..., rL]

        Returns:
            quantized: numpy array, shape (n_vectors, dimension)
                      重建的量化向量
        """
        if not self.codebooks:
            raise ValueError("Codebooks未训练! 请先调用train()方法。")

        n_vectors = codes.shape[0]
        quantized = np.zeros((n_vectors, self.dimension), dtype=np.float32)

        # 逐层累加中心点
        for level in range(self.n_levels):
            level_codes = codes[:, level]
            level_centroids = self.codebooks[level][level_codes]
            quantized += level_centroids

        return quantized

    def save(self, path):
        # 确保path以.npz结尾
        if not path.endswith('.npz'):
            path = path + '.npz'

        # 确保保存目录存在（与 PSRQ 预模型保存逻辑对齐）
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # 分别保存每层codebook，避免dtype=object的序列化问题
        save_dict = {
            'n_levels': self.n_levels,
            'codebook_size': self.codebook_size,
            'dimension': self.dimension,
            'random_state': self.random_state
        }
        for i, cb in enumerate(self.codebooks):
            save_dict[f'codebook_{i}'] = cb
        
        np.savez(path, **save_dict)
        print(f"Codebooks已保存到 {path}")

    def load(self, path):
        # 确保path以.npz结尾
        if not path.endswith('.npz'):
            path = path + '.npz'
        
        data = np.load(path, allow_pickle=False)  # 避免pickle安全风险
        self.n_levels = int(data['n_levels'])
        self.codebook_size = int(data['codebook_size'])
        self.dimension = int(data['dimension'])
        self.random_state = int(data.get('random_state', 42))
        
        # 加载各层codebook
        self.codebooks = []
        for i in range(self.n_levels):
            self.codebooks.append(data[f'codebook_{i}'])

        print(f"Codebooks已从 {path} 加载. 层数: {self.n_levels}, 每层大小: {self.codebook_size}")
