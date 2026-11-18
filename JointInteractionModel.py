import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN import MultiViewDrugEncoder, CONFIG as DRUG_CONFIG
from protein_train_cl3 import SiameseInteractionEncoder
from interaction_model5 import MutationDrugInteractionModel


def contrastive_loss_fn(z1, z2, temp=0.1):
    """用于计算药物多视图之间的对比损失"""

    z1 = F.normalize(z1, p=2, dim=1, eps=1e-4)
    z2 = F.normalize(z2, p=2, dim=1, eps=1e-4)
    # z1和z2是药物在不同试图下提取的特征向量
    sim_matrix = torch.matmul(z1, z2.T) / temp
    sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)

    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim_matrix, labels)


class JointInteractionModel(nn.Module):
    """
        药物-蛋白质联合交互模型：
        核心流程：
        1. 药物多视图编码 + 药物对比损失计算
        2. 蛋白质孪生编码 + 蛋白质对比损失计算
        3. 药物-蛋白质交互预测（基于注意力的跨模态交互）
        """
    def __init__(self, brics_vocab_size, drug_config=DRUG_CONFIG,
                 prot_struct_dim=512, prot_seq_dim=1280, embed_dim=256, temp=0.07):
        """
                初始化联合模型
                :param brics_vocab_size: 药物BRICS子结构词表大小
                :param drug_config: 药物编码器配置（默认使用GNN.py的DRUG_CONFIG）
                :param prot_struct_dim: 蛋白质结构特征维度（默认512）
                :param prot_seq_dim: 蛋白质序列特征维度（默认1280）
                :param embed_dim: 特征融合后的统一维度（默认256）
                :param temp: 蛋白质对比损失的温度系数（默认0.07）
                """
        super().__init__()

        #  初始化药物多视图编码器（核心：三视图编码+门控融合)
        self.drug_encoder = MultiViewDrugEncoder(drug_config, brics_vocab_size)

        # 2. 初始化蛋白质孪生编码器（处理突变/野生型蛋白的结构+序列特征，计算蛋白对比损失）
        self.prot_encoder = SiameseInteractionEncoder(
            struct_dim=prot_struct_dim,
            seq_dim=prot_seq_dim,
            embed_dim=embed_dim,
            temp=temp
        )

        # 3. 初始化药物-蛋白质交互预测器（基于多头注意力的跨模态交互建模
        self.interaction_model = MutationDrugInteractionModel(
            hidden_dim=embed_dim,
            num_heads=4,
            dropout=0.2
        )

    def forward(self, drug_batch, mut_s, mut_q, wild_s, wild_q, prot_mask):
        """
                前向传播：输入药物+蛋白质特征，输出交互预测结果+对比损失
                :param drug_batch: 药物批数据（GNN.py的multiview_collate_fn输出）
                :param mut_s: 突变型蛋白结构特征 [batch_size, seq_len, prot_struct_dim]
                :param mut_q: 突变型蛋白序列特征 [batch_size, seq_len, prot_seq_dim]
                :param wild_s: 野生型蛋白结构特征 [batch_size, seq_len, prot_struct_dim]
                :param wild_q: 野生型蛋白序列特征 [batch_size, seq_len, prot_seq_dim]
                :param prot_mask: 蛋白质序列Padding掩码 [batch_size, seq_len]（True=有效位置，False=PAD）
                :return: 字典形式的输出：
                    - logits: 药物-蛋白质交互预测值 [batch_size, seq_len]
                    - cl_loss_prot: 蛋白质对比损失（标量）
                    - cl_loss_drug: 药物对比损失（标量）
                """
        # ==========================================
        # 1. 动态提取药物特征 & 计算药物 CL Loss
        # ==========================================
        drug_out = self.drug_encoder(drug_batch)
        drug_fused = drug_out['fused']  # [Batch, 256]

        # 计算药物多视图对比损失
        views_proj = drug_out['views_proj']
        loss_cl_drug_1 = contrastive_loss_fn(views_proj[0], views_proj[1])
        loss_cl_drug_2 = contrastive_loss_fn(views_proj[1], views_proj[2])
        cl_loss_drug =0.5 *  loss_cl_drug_1 + 0.5 * loss_cl_drug_2

        # ==========================================
        # 2. 动态提取蛋白特征 & 计算蛋白 CL Loss
        # ==========================================
        prot_out = self.prot_encoder(mut_s, mut_q, wild_s, wild_q, prot_mask)
        prot_fused = prot_out['fused_embedding']  # [Batch, Seq_Len, 512]
        cl_loss_prot = prot_out['cl_loss']

        # ==========================================
        # 3. 相互作用预测
        # ==========================================

        bs = drug_fused.size(0)
        drug_emb_seq = drug_fused.unsqueeze(1)  # [Batch, 1, 256]
        drug_mask = torch.ones((bs, 1), dtype=torch.bool, device=drug_fused.device)

        # 送入交互预测器
        logits = self.interaction_model(prot_fused, drug_emb_seq, prot_mask, drug_mask)

        return {
            'logits': logits,
            'cl_loss_prot': cl_loss_prot,
            'cl_loss_drug': cl_loss_drug
        }