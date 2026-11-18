import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==============================================================================
# 第一部分：Siamese Dataset
# ==============================================================================

class SiameseProteinDataset(Dataset):
    def __init__(self, mut_gvp_dir, mut_esm_dir, wild_gvp_dir, wild_esm_dir):
        self.mut_gvp_dir = mut_gvp_dir
        self.mut_esm_dir = mut_esm_dir
        self.wild_gvp_dir = wild_gvp_dir
        self.wild_esm_dir = wild_esm_dir

        self.data_pairs = []


        mut_files = [f for f in os.listdir(mut_gvp_dir) if f.endswith('_pocket.pt')]
        print(f"扫描数据中... 基于 Mut-GVP 文件数: {len(mut_files)}")

        missing_count = 0
        for mut_gvp_file in mut_files:
            base_name = mut_gvp_file.replace("_pocket.pt", "")

            # 构造文件名
            mut_esm_file = f"{base_name}.pt"
            wild_gvp_file = f"{base_name}_WT_pocket.pt"
            wild_esm_file = f"{base_name}_WT.pt"

            # 检查文件
            p1 = os.path.join(mut_esm_dir, mut_esm_file)
            p2 = os.path.join(wild_gvp_dir, wild_gvp_file)
            p3 = os.path.join(wild_esm_dir, wild_esm_file)

            if os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3):
                self.data_pairs.append({
                    'name': base_name,
                    'mut_gvp': mut_gvp_file,
                    'mut_esm': mut_esm_file,
                    'wild_gvp': wild_gvp_file,
                    'wild_esm': wild_esm_file
                })
            else:
                missing_count += 1

        print(f"成功配对完整四元组: {len(self.data_pairs)} (丢弃: {missing_count})")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]

        # 1. 加载数据字典 (包含 'embedding' 和 'ids')
        d_mut_s = torch.load(os.path.join(self.mut_gvp_dir, item['mut_gvp']), map_location='cpu')
        d_mut_q = torch.load(os.path.join(self.mut_esm_dir, item['mut_esm']), map_location='cpu')
        d_wild_s = torch.load(os.path.join(self.wild_gvp_dir, item['wild_gvp']), map_location='cpu')
        d_wild_q = torch.load(os.path.join(self.wild_esm_dir, item['wild_esm']), map_location='cpu')

        # 2. 确定锚点 (Anchor): 以 Mut Structure 为基准
        anchor_ids = d_mut_s['ids']

        # 3. 辅助函数：构建 ID -> Vector 的查找表
        def make_lookup(data_dict):
            return {uid: vec for uid, vec in zip(data_dict['ids'], data_dict['embedding'])}

        dict_mut_q = make_lookup(d_mut_q)
        dict_wild_s = make_lookup(d_wild_s)
        dict_wild_q = make_lookup(d_wild_q)

        # 4. 对齐逻辑
        aligned_mut_s = d_mut_s['embedding']  # 锚点直接用
        aligned_mut_q = []
        aligned_wild_s = []
        aligned_wild_q = []

        # 获取维度用于补零
        struct_dim = aligned_mut_s.shape[1]
        seq_dim = d_mut_q['embedding'].shape[1]

        for uid in anchor_ids:
            # --- 对齐 Mut Sequence ---
            if uid in dict_mut_q:
                aligned_mut_q.append(dict_mut_q[uid])
            else:
                aligned_mut_q.append(torch.zeros(seq_dim))

            # --- 对齐 Wild Structure ---
            if uid in dict_wild_s:
                aligned_wild_s.append(dict_wild_s[uid])
            else:
                aligned_wild_s.append(torch.zeros(struct_dim))

            # --- 对齐 Wild Sequence ---
            if uid in dict_wild_q:
                aligned_wild_q.append(dict_wild_q[uid])
            else:
                aligned_wild_q.append(torch.zeros(seq_dim))

        # 5. 堆叠并返回
        return (
            aligned_mut_s.float(),
            torch.stack(aligned_mut_q).float(),
            torch.stack(aligned_wild_s).float(),
            torch.stack(aligned_wild_q).float(),
            item['name']
        )


def collate_fn_siamese(batch):
    mut_s, mut_q, wild_s, wild_q, names = zip(*batch)

    # 记录真实长度
    lengths = torch.tensor([x.size(0) for x in mut_s])

    # Padding
    mut_s_pad = pad_sequence(mut_s, batch_first=True, padding_value=0.0)
    mut_q_pad = pad_sequence(mut_q, batch_first=True, padding_value=0.0)
    wild_s_pad = pad_sequence(wild_s, batch_first=True, padding_value=0.0)
    wild_q_pad = pad_sequence(wild_q, batch_first=True, padding_value=0.0)

    # 生成 Mask
    max_len = mut_s_pad.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return mut_s_pad, mut_q_pad, wild_s_pad, wild_q_pad, mask


# ==============================================================================
# 第二部分：SiameseInteractionEncoder
# ==============================================================================

class SiameseInteractionEncoder(nn.Module):
    def __init__(self, struct_dim=512, seq_dim=1280, embed_dim=256, temp=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temp = temp

        # 1. 独立投影层（把他们投影到相同的维度）
        self.proj_struct = nn.Sequential(
            nn.Linear(struct_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.proj_seq = nn.Sequential(
            nn.Linear(seq_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 2. 学习型融合层 (Learned Fusion)
        self.fusion_struct = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.fusion_seq = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, mut_s, mut_q, wild_s, wild_q, mask):
        # 1. 投影
        h_mut_s = self.proj_struct(mut_s)  # 突变结构
        h_wild_s = self.proj_struct(wild_s)  # 野生结构
        h_mut_q = self.proj_seq(mut_q)  # 突变序列
        h_wild_q = self.proj_seq(wild_q)  # 野生序列

        # 2. 结构融合
        cat_struct = torch.cat([h_mut_s, h_wild_s], dim=-1)  # 拼接
        z_struct_diff = self.fusion_struct(cat_struct)  # 学习两者之间的差异

        # 3. 序列融合
        cat_seq = torch.cat([h_mut_q, h_wild_q], dim=-1)  # 拼接
        z_seq_diff = self.fusion_seq(cat_seq)  # 学习两者之间的差异

        # 4. Masking
        mask_unsqueeze = mask.unsqueeze(-1).float()
        z_struct_diff = z_struct_diff * mask_unsqueeze
        z_seq_diff = z_seq_diff * mask_unsqueeze
        h_mut_s = h_mut_s * mask_unsqueeze

        # 5. 计算 InfoNCE Loss
        loss = self.compute_contrastive_loss(z_struct_diff, z_seq_diff, mask)

        # 6. 返回结果
        final_embedding = torch.cat([h_mut_s, z_struct_diff], dim=-1)  # 返回的结果是两个，第一个是突变型蛋白质口袋的绝对物理环境信息，第二个是从野生型到突变型的动态变化

        return {
            'fused_embedding': final_embedding,
            'cl_loss': loss
        }

    def compute_contrastive_loss(self, z1, z2, mask):
        valid_indices = mask.view(-1).bool()
        z1_flat = z1.reshape(-1, self.embed_dim)[valid_indices]
        z2_flat = z2.reshape(-1, self.embed_dim)[valid_indices]

        if z1_flat.shape[0] == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        z1_norm = F.normalize(z1_flat, dim=1)
        z2_norm = F.normalize(z2_flat, dim=1)

        logits = torch.matmul(z1_norm, z2_norm.T) / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


