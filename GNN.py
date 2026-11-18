import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

# ==========================================
# 引入 PyTorch Geometric (PyG) 依赖
# ==========================================
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool

from config import DRUG_CSV

# ==========================================
# 第一部分：配置与工具函数
# ==========================================

CONFIG = {
    'csv_path': str(DRUG_CSV),
    'd_model': 256,
    'fp_dim': 2048,
    'brics_min_freq': 1

}


def safe_read_csv(file_path):
    """安全读取 CSV，自动处理不同的编码格式"""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')


# ==========================================
# 第二部分：数据处理与加载器
# ==========================================

class MoleculeProcessor:
    """处理 SMILES 并转换为模型所需的三个视图特征 (引入 PyG 结构)"""
    """
    初始化处理器
    param brics_vocab: BRICS子结构词表（{子结构SMILES: 索引}），默认包含<PAD>和<UNK>
    param fp_dim: Morgan指纹维度，默认2048
    """
    def __init__(self, brics_vocab=None, fp_dim=2048):
        self.brics_vocab = brics_vocab if brics_vocab else {'<PAD>': 0, '<UNK>': 1}
        self.fp_dim = fp_dim
        # 定义关注的原子类型（共9种，覆盖药物分子常见原子）
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I']

    def get_brics_seq(self, smiles):
        """
        View1 - 序列视角：提取BRICS子结构序列
        核心逻辑：用BRICS算法拆分分子为化学意义的子结构，转化为词表索引序列
        :param smiles: 药物分子的SMILES字符串
        :return: 子结构索引序列（List[int]），异常时返回[0]（PAD）
        """
        mol = Chem.MolFromSmiles(smiles)  # 解析SMILES为RDKit分子对象
        if mol is None: return [0]  # 分子解析失败，返回PAD序列
        try:
            # BRICS算法拆分分子（打破BRICS键）
            broken = BRICS.BreakBRICSBonds(mol)
            # 提取拆分后的子结构（作为独立分子对象）
            frags = Chem.GetMolFrags(broken, asMols=True)
            # 子结构转化为SMILES字符串（保留异构信息）
            frag_smiles = [Chem.MolToSmiles(f, isomericSmiles=True) for f in frags]
            # 子结构映射到词表索引：存在则取索引，否则取<UNK>（1）
            seq = [self.brics_vocab.get(f, 1) for f in frag_smiles]
            # 空序列兜底（拆分后无结果）
            if len(seq) == 0: return [1]
            return seq
        except:
            return [1]

    def get_graph_features(self, smiles):
        """View 2: Graph (使用 PyG 的 edge_index 和稀疏特征表示)"""
        """
            View2 - 图拓扑视角：提取分子的图结构特征（节点特征+边索引）
            核心逻辑：原子→节点（One-Hot编码原子类型），化学键→边（无向图，双向边）
            :param smiles: 药物分子的SMILES字符串
            :return: 
                node_feats: 节点特征矩阵 [num_atoms, 9]（9为原子类型数），float32
                edge_index: 边索引矩阵 [2, num_edges]（无向图，每行是源/目标节点索引），int64
        """
        # 解析SMILES为RDKit分子对象
        mol = Chem.MolFromSmiles(smiles)

        # 异常处理：如果分子无效，返回包含 1 个虚拟节点和 0 条边的图
        if mol is None:
            return (np.zeros((1, len(self.atom_types)), dtype=np.float32),
                    np.empty((2, 0), dtype=np.int64))

        num_atoms = mol.GetNumAtoms()  # 原子数量（节点数）

        # 1. 提取节点特征 x: [num_nodes, num_features] 构建节点特征：One-Hot编码原子类型
        node_feats = np.zeros((num_atoms, len(self.atom_types)), dtype=np.float32)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            if symbol in self.atom_types:
                node_feats[i, self.atom_types.index(symbol)] = 1.0

        # 2. 提取边索引 edge_index: [2, num_edges] 构建边索引：无向图，每条边添加双向连接
        src, dst = [], []  # 源节点列表、目标节点列表
        for bond in mol.GetBonds():  # 遍历所有化学键
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # 边的起始/终止原子索引
            src.extend([i, j])
            dst.extend([j, i])  # 无向图，添加双向边

        # 转化为numpy数组：无边缘时返回空数组
        edge_index = np.array([src, dst], dtype=np.int64) if src else np.empty((2, 0), dtype=np.int64)

        return node_feats, edge_index

    def get_fingerprint(self, smiles):
        """
        View3 - 指纹视角：提取2048维Morgan指纹（圆形指纹）
        核心逻辑：基于Morgan算法生成分子的固定维度二进制指纹，表征分子整体结构
        :param smiles: 药物分子的SMILES字符串
        :return: 2048维指纹数组（float32），异常时返回全0
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(self.fp_dim, dtype=np.float32)
        # 生成Morgan指纹：半径2，维度2048，返回二进制位向量
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_dim)
        return np.array(fp, dtype=np.float32)


class MultiViewDrugDataset(Dataset):
    def __init__(self, csv_path, brics_vocab, fp_dim=2048):
        self.df = safe_read_csv(csv_path)
        self.processor = MoleculeProcessor(brics_vocab, fp_dim=fp_dim)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = str(row['SMILES'])
        # 提取三视图特征
        brics_seq = self.processor.get_brics_seq(smiles)
        node_feats, edge_index = self.processor.get_graph_features(smiles)
        fp = self.processor.get_fingerprint(smiles)

        # 构建 PyG 的 Data 对象
        graph_data = Data(
            x=torch.FloatTensor(node_feats),
            edge_index=torch.LongTensor(edge_index)
        )

        return {
            'brics_seq': torch.LongTensor(brics_seq),
            'graph_data': graph_data,
            'fingerprint': torch.FloatTensor(fp)
        }


def multiview_collate_fn(batch):
    """自定义整理函数，处理变长序列和 PyG 图的批处理"""
    """
        自定义批处理函数（Collate Function）：处理变长序列和PyG图数据的批拼接
        核心解决：
            1. BRICS序列是变长的，需要Padding到同长度
            2. PyG图数据需要用Batch.from_data_list拼接为批图（对角分块大图）
        :param batch: 单样本特征字典的列表
        :return: 批处理后的特征字典：
            - brics_seq: 填充后的序列 [batch_size, max_seq_len]
            - graph_batch: 批图数据（PyG Batch对象）
            - fingerprint: 指纹批 [batch_size, 2048]
        """
    # 1. 序列填充
    brics_seqs = [item['brics_seq'] for item in batch]
    brics_padded = torch.nn.utils.rnn.pad_sequence(brics_seqs, batch_first=True, padding_value=0)

    # 2. PyG 图批处理 (自动拼成对角分块大图，彻底抛弃 Padding 和 Mask)
    graph_data_list = [item['graph_data'] for item in batch]
    graph_batch = Batch.from_data_list(graph_data_list)

    return {
        'brics_seq': brics_padded,
        'graph_batch': graph_batch,
        'fingerprint': torch.stack([item['fingerprint'] for item in batch])
    }


def build_brics_vocab(csv_path):
    """
        构建BRICS子结构词表：统计数据集中所有SMILES的BRICS子结构，过滤低频子结构
        :param csv_path: 包含SMILES列的CSV文件路径
        :return: 构建好的词表 {子结构SMILES: 索引}
        """
    print(f"正在扫描数据集构建 BRICS 词表: {csv_path}...")
    df = safe_read_csv(csv_path)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    frag_counts = {}

    for sm in df['SMILES'].dropna():
        mol = Chem.MolFromSmiles(sm)
        if mol:
            try:
                broken = BRICS.BreakBRICSBonds(mol)
                frags = Chem.GetMolFrags(broken, asMols=True)
                for f in frags:
                    s = Chem.MolToSmiles(f, isomericSmiles=True)
                    frag_counts[s] = frag_counts.get(s, 0) + 1
            except:
                pass

    for frag, count in frag_counts.items():
        if count >= CONFIG['brics_min_freq']:
            vocab[frag] = idx
            idx += 1

    print(f"词表构建完成，大小: {len(vocab)}")
    return vocab


# ==========================================
# 第三部分：模型架构
# ==========================================

class SubstructureEncoder(nn.Module):
    """
        View1编码器：子结构序列编码器（基于Transformer）
        功能：将变长BRICS序列编码为固定维度（d_model）的序列级特征
        """
    def __init__(self, vocab_size, embed_dim):
        """
        初始化序列编码器
        :param vocab_size: BRICS词表大小
        :param embed_dim: 嵌入维度（最终输出特征维度，对应d_model）
        """
        super().__init__()
        # 词嵌入层：将子结构索引转化为向量，padding_idx=0（PAD不参与梯度更新）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Transformer编码器：自注意力捕捉子结构间依赖
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=512),
            num_layers=2
        )

    def forward(self, x):
        """
                前向传播：序列→固定维度特征
                :param x: 填充后的BRICS序列 [batch_size, max_seq_len]
                :return: 序列级特征 [batch_size, embed_dim]（Masked Mean Pooling）
                """
        mask = (x == 0)
        # 词嵌入：[batch_size, max_seq_len, embed_dim]
        emb = self.embedding(x)
        # Transformer编码：屏蔽PAD位置，输出 [batch_size, max_seq_len, embed_dim]
        feat = self.transformer(emb, src_key_padding_mask=mask)
        # Masked Mean Pooling：只对非PAD位置求平均，避免PAD影响
        mask_float = (~mask).float().unsqueeze(-1)
        sum_feat = torch.sum(feat * mask_float, dim=1)  # 非PAD位置特征求和
        count = torch.sum(mask_float, dim=1).clamp(min=1e-4)  # 非PAD位置数（防止除0）
        return sum_feat / count  # 平均池化：[batch_size, embed_dim]


class GraphEncoder(nn.Module):
    """
        View2编码器：图结构编码器
        功能：将分子图（节点+边）编码为固定维度（hidden_dim）的图级特征
        """
    def __init__(self, input_dim, hidden_dim):
        """
        初始化图编码器
        :param input_dim: 节点特征维度（9，对应9种原子类型）
        :param hidden_dim: 隐藏层/输出维度（对应d_model）
        """
        super().__init__()
        # 两层GCN卷积：PyG原生GCNConv，适配稀疏图计算
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, return_nodes=False):
        """
        前向传播：分子图→图级特征/节点特征
        :param x: 节点特征 [total_nodes, input_dim]（批内所有图的节点拼接）
        :param edge_index: 边索引 [2, total_edges]（批内所有图的边拼接）
        :param batch: 批索引 [total_nodes]（标记每个节点属于哪个图）
        :param return_nodes: 是否返回节点级特征（默认False，返回图级特征）
        :return: 图级特征 [batch_size, hidden_dim] 或 节点级特征 [total_nodes, hidden_dim]
        """
        # 第一层GCN：消息传递 + 激活
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        # 第二层GCN：进一步捕捉高阶拓扑信息
        x = self.gcn2(x, edge_index)

        if return_nodes:
            return x

        # 全局最大池化：按batch索引聚合每个图的节点特征，得到图级特征
        # 优势：无需Padding，自动适配不同大小的图，避免掩码溢出问题
        h_graph = global_max_pool(x, batch)
        return h_graph


class FingerprintEncoder(nn.Module):
    """
    View3编码器：指纹特征编码器（基于MLP）
    功能：将2048维Morgan指纹降维为固定维度（output_dim）的指纹特征
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 多层感知机：非线性降维，加入Dropout防止过拟合
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        """
                前向传播：指纹→固定维度特征
                :param x: Morgan指纹 [batch_size, 2048]
                :return: 指纹特征 [batch_size, output_dim]
                """
        return self.mlp(x)


class GatedFusion(nn.Module):
    """
        门控多视图融合模块：基于注意力机制自动分配各视图权重，融合为最终特征
        核心优势：相比平均/拼接，能根据分子特性动态调整视图权重（如大分子侧重图，对称分子侧重指纹）
        """
    def __init__(self, embed_dim):
        super().__init__()
        # 注意力评分网络：输入单视图特征，输出注意力分数
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, views):
        """
        前向传播：融合多视图特征
        :param views: 三视图特征列表 [h_sub, h_graph, h_fp]，每个形状[batch_size, embed_dim]
        :return:
            h_fused: 融合后的特征 [batch_size, embed_dim]
            alpha: 各视图注意力权重 [batch_size, 3, 1]
        """

        stack = torch.stack(views, dim=1)
        # 计算注意力评分：[batch_size, 3, 1]
        scores = self.attn_net(stack)
        # Softmax归一化：得到各视图权重（和为1）
        alpha = F.softmax(scores, dim=1)
        # 加权求和：融合特征 = 权重1*视图1 + 权重2*视图2 + 权重3*视图3
        h_fused = torch.sum(alpha * stack, dim=1)
        return h_fused, alpha


class MultiViewDrugEncoder(nn.Module):
    """
    多视图药物编码器：整合三视图编码器 + 门控融合，输出最终药物表征
    """
    def __init__(self, config, brics_vocab_size):
        super().__init__()
        self.sub_enc = SubstructureEncoder(brics_vocab_size, config['d_model'])
        self.graph_enc = GraphEncoder(len(['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I']), config['d_model'])
        self.fp_enc = FingerprintEncoder(config['fp_dim'], config['d_model'])
        self.fusion = GatedFusion(config['d_model'])

        self.proj_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )

    def forward(self, batch, extract_atom_seq=False):
        # 1. 序列
        h_sub = self.sub_enc(batch['brics_seq'])

        # 2. 图 (提取 PyG Batch)
        graph_batch = batch['graph_batch']
        h_graph_global = self.graph_enc(graph_batch.x, graph_batch.edge_index, graph_batch.batch, return_nodes=False)

        # 3. 指纹
        h_fp = self.fp_enc(batch['fingerprint'])
        # 4. 门控融合三视图特征
        views = [h_sub, h_graph_global, h_fp]
        h_fused, weights = self.fusion(views)
        # 5. 投影头：将各视图特征投影到同一隐空间（用于对比学习）
        views_proj = [self.proj_head(v) for v in views]

        output = {
            'fused': h_fused,
            'views': views,
            'views_proj': views_proj,
            'weights': weights
        }

        if extract_atom_seq:
            atom_features = self.graph_enc(graph_batch.x, graph_batch.edge_index, graph_batch.batch, return_nodes=True)
            output['atom_features'] = atom_features

        return output


def contrastive_loss(z1, z2, temp=0.1):
    """
        对比损失函数：拉近同一样本不同视图的特征距离，拉远不同样本的特征距离
        核心：InfoNCE损失，用于多视图对比学习
        :param z1: 视图1的投影特征 [batch_size, d_model]
        :param z2: 视图2的投影特征 [batch_size, d_model]
        :param temp: 温度系数（控制分布平滑度）
        :return: 对比损失值（标量）
        """
    z1 = F.normalize(z1, p=2, dim=1, eps=1e-8)
    z2 = F.normalize(z2, p=2, dim=1, eps=1e-8)
    sim_matrix = torch.matmul(z1, z2.T) / temp
    sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.CrossEntropyLoss()(sim_matrix, labels)

