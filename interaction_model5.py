import torch
import torch.nn as nn
import torch.nn.functional as F


class MutationDrugInteractionModel(nn.Module):
    """
        突变-药物相互作用预测模型
        核心思路：
        1. 对蛋白（含Context/Delta双流）和药物特征做投影对齐维度
        2. 多头注意力捕捉药物-蛋白的双向交互
        3. 低秩双线性池化捕捉特征乘性交互，结合Wide&Deep结构做最终预测
        """
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.2):
        """
        模型初始化函数
        Args:
            hidden_dim: 隐藏层维度（核心特征维度），默认256
            num_heads: 多头注意力的头数，默认4
            dropout: 随机失活概率，默认0.2（防止过拟合）
        """
        super().__init__()
        self.hidden_dim = hidden_dim  # 保存隐藏层维度为实例变量

        # ==========================================================
        # 1. 双流输入投影模块：将原始输入映射到统一的隐藏维度
        #  蛋白输入拆分为Context（前256维）和Delta（后256维），药物单独投影
        # ==========================================================
        # 输入维度固定 512 (前256=Context, 后256=Delta)
        self.proj_context = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_delta = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_drug = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # ==========================================================
        # 2. 交互注意力模块：多头注意力捕捉特征间的依赖关系
        # MultiheadAttention参数说明：
        # - embed_dim: 输入特征维度（需与hidden_dim一致）
        # - num_heads: 注意力头数（需能被embed_dim整除）
        # - dropout: 注意力层内的Dropout
        # - batch_first: 输入格式为[batch_size, seq_len, embed_dim]（符合PyTorch常用格式）
        # ==========================================================
        # 2.1 Drug→Context注意力：药物特征查询，蛋白Context特征作为键值
        self.attn_drug_context = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # 2.2 Drug→Delta注意力：药物特征查询，蛋白Delta特征作为键值
        self.attn_drug_delta = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # 2.3 Prot→Drug注意力：蛋白Context特征查询，药物特征作为键值
        self.attn_prot_drug = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # ==========================================================
        # 3. 低秩双线性分类头：捕捉特征间的乘性交互，同时保留原始特征（Wide&Deep）
        # ==========================================================


        # 3.1 特征分组融合层
        # Group A (Drug Side): 将 Binding + Mutation + Raw 融合为一个向量 V_drug
        self.drug_group_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Group B (Prot Side): 对 Prot 特征进行映射，确保语义空间对齐 V_prot
        self.prot_group_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3.2 双线性映射 (Bilinear Map)
        # 我们使用 Hadamard Product + Linear 来近似 Bilinear Pooling
        # 这捕捉了 V_drug 和 V_prot 的乘性交互
        self.bilinear_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3.3 最终预测层 (Wide & Deep 结构)
        # 输入: [Bilinear_Feature; V_drug; V_prot]
        # 维度: hidden_dim * 3
        # 这种残差结构保证了模型既能学到交互，又不会丢失原始信息
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, prot_emb, drug_emb, prot_mask, drug_mask):
        """
        模型前向传播函数（核心执行逻辑）
        Args:
            prot_emb: 蛋白嵌入特征，shape=[batch_size, prot_seq_len, 512]（前256=Context，后256=Delta）
            drug_emb: 药物嵌入特征，shape=[batch_size, drug_seq_len, hidden_dim]
            prot_mask: 蛋白序列掩码，shape=[batch_size, prot_seq_len]（True=有效位置，False=填充位置）
            drug_mask: 药物序列掩码，shape=[batch_size, drug_seq_len]（True=有效位置，False=填充位置）
        Returns:
            logits: 预测值，shape=[batch_size, 1]
        """
        # 1. 维度检查与切分
        if prot_emb.shape[-1] != 512:
            raise ValueError(f"Input mismatch! Expected 512, got {prot_emb.shape[-1]}")

        raw_context = prot_emb[:, :, :256]
        raw_delta = prot_emb[:, :, 256:]

        # 2. 投影
        h_ctx = self.proj_context(raw_context)
        h_delta = self.proj_delta(raw_delta)
        h_drug = self.proj_drug(drug_emb)

        attn_pad_mask_prot = ~prot_mask
        attn_pad_mask_drug = ~drug_mask

        # 3. 注意力交互
        # A. Drug -> Context  注意力层的输入遵循query=目标特征，key/value=参考特征，输出为 “融合参考特征的目标特征”
        context_aware_drug, _ = self.attn_drug_context(
            query=h_drug, key=h_ctx, value=h_ctx, key_padding_mask=attn_pad_mask_prot
        )
        # B. Drug -> Delta
        delta_aware_drug, _ = self.attn_drug_delta(
            query=h_drug, key=h_delta, value=h_delta, key_padding_mask=attn_pad_mask_prot
        )
        # C. Prot -> Drug (Context as Prot)
        drug_aware_prot, _ = self.attn_prot_drug(
            query=h_ctx, key=h_drug, value=h_drug, key_padding_mask=attn_pad_mask_drug
        )


        # ==========================================================
        # 4. 全局池化：将序列特征（seq_len维度）聚合为单个向量
        # 掩码转换为浮点型，用于加权求和（填充位置权重为0）
        drug_mask_f = drug_mask.unsqueeze(-1).float()
        prot_mask_f = prot_mask.unsqueeze(-1).float()

        # 4. 聚合 (Global Pooling)
        # 4.1 药物侧聚合：Context感知的药物特征 → 全局向量
        feat_binding = (context_aware_drug * drug_mask_f).sum(1) / drug_mask_f.sum(1).clamp(min=1e-9)
        # 4.2 药物侧聚合：Delta感知的药物特征 → 全局向量
        feat_mutation = (delta_aware_drug * drug_mask_f).sum(1) / drug_mask_f.sum(1).clamp(min=1e-9)
        # 4.3 药物侧聚合：原始药物特征 → 全局向量
        feat_drug_raw = (h_drug * drug_mask_f).sum(1) / drug_mask_f.sum(1).clamp(min=1e-9)
        # 4.4 蛋白侧聚合：药物感知的蛋白特征 → 全局向量（clamp防止除以0)
        feat_prot = (drug_aware_prot * prot_mask_f).sum(1) / prot_mask_f.sum(1).clamp(min=1e-9)
        # 注：sum(1)是对seq_len维度求和，除以有效长度（mask.sum(1)）实现均值池化

        # 5. 特征分组与双线性交互 (Bilinear Pooling)
        # 5.1 构造 V_drug (融合药物侧的三种视角)
        v_drug_raw_concat = torch.cat([feat_binding, feat_mutation, feat_drug_raw], dim=-1)  # [B, 3*H]
        v_drug = self.drug_group_fusion(v_drug_raw_concat)  # [B, H]

        # 5.2 构造 V_prot
        v_prot = self.prot_group_fusion(feat_prot)  # [B, H]

        # ==========================================================
        # 双线性相乘前的 L2 归一化防溢出
        # ==========================================================
        v_drug_norm = F.normalize(v_drug, p=2, dim=1, eps=1e-6)
        v_prot_norm = F.normalize(v_prot, p=2, dim=1, eps=1e-6)

        # 5.3 核心交互: Hadamard Product (逐元素积)
        bilinear_feat = v_drug_norm * v_prot_norm  # [B, H]
        bilinear_feat = F.normalize(bilinear_feat, p=2, dim=1, eps=1e-4)
        bilinear_out = self.bilinear_proj(bilinear_feat)  # [B, H]

        # 6. 残差连接与最终预测
        # 拼接: [交互特征; 原始药物特征; 原始蛋白特征]
        final_vec = torch.cat([bilinear_out, v_drug, v_prot], dim=-1)  # [B, 3*H]

        logits = self.predictor(final_vec)

        return logits

