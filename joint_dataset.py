import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 引入 PyTorch Geometric 的数据结构
from torch_geometric.data import Data, Batch

from GNN import MoleculeProcessor, safe_read_csv


class JointInteractionDataset(Dataset):
    def __init__(self, csv_path, brics_vocab,
                 drug_config, prot_dirs):
        """
        prot_dirs 应该是一个字典:
        {'mut_gvp': '...', 'mut_esm': '...', 'wild_gvp': '...', 'wild_esm': '...'}
        """
        self.df = safe_read_csv(csv_path)

        # 1. 适配 PyG 版的 GNN
        self.processor = MoleculeProcessor(
            brics_vocab, fp_dim=drug_config['fp_dim']
        )
        self.prot_dirs = prot_dirs

        # 预先过滤掉文件缺失的数据对
        self.valid_data = []
        for idx, row in self.df.iterrows():
            pdb_id = str(row.get('PDB_ID', row.iloc[0])).strip()
            drug_id = str(row.get('Drug_ID', row.iloc[1])).strip()
            smiles = str(row['SMILES']) if 'SMILES' in row else None  # 需要确保 CSV 里有 SMILES
            label = float(row.get('Label', row.get('Relation', 0.0)))

            # 检查蛋白质文件是否存在
            m_gvp = os.path.join(prot_dirs['mut_gvp'], f"{pdb_id}_pocket.pt")
            m_esm = os.path.join(prot_dirs['mut_esm'], f"{pdb_id}.pt")
            w_gvp = os.path.join(prot_dirs['wild_gvp'], f"{pdb_id}_WT_pocket.pt")
            w_esm = os.path.join(prot_dirs['wild_esm'], f"{pdb_id}_WT.pt")

            if all(os.path.exists(p) for p in [m_gvp, m_esm, w_gvp, w_esm]) and smiles:
                self.valid_data.append({
                    'pdb_id': pdb_id, 'smiles': smiles, 'label': label,
                    'paths': (m_gvp, m_esm, w_gvp, w_esm)
                })

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]

        # 1. 药物原始特征处理 (PyG 适配版)
        brics_seq = self.processor.get_brics_seq(item['smiles'])

        # 接收 PyG 所需的 node_feats 和 edge_index
        node_feats, edge_index = self.processor.get_graph_features(item['smiles'])
        fp = self.processor.get_fingerprint(item['smiles'])

        # 构建 PyG 的 Data 对象
        graph_data = Data(
            x=torch.FloatTensor(node_feats),
            edge_index=torch.LongTensor(edge_index)
        )

        # 返回新的药物字典结构
        drug_data = {
            'brics_seq': torch.LongTensor(brics_seq),
            'graph_data': graph_data,
            'fingerprint': torch.FloatTensor(fp)
        }

        # 2. 蛋白质原始特征处理 (保持对齐逻辑不变)
        m_gvp, m_esm, w_gvp, w_esm = item['paths']
        d_mut_s = torch.load(m_gvp, map_location='cpu')
        d_mut_q = torch.load(m_esm, map_location='cpu')
        d_wild_s = torch.load(w_gvp, map_location='cpu')
        d_wild_q = torch.load(w_esm, map_location='cpu')

        anchor_ids = d_mut_s['ids']
        dict_mut_q = {uid: vec for uid, vec in zip(d_mut_q['ids'], d_mut_q['embedding'])}
        dict_wild_s = {uid: vec for uid, vec in zip(d_wild_s['ids'], d_wild_s['embedding'])}
        dict_wild_q = {uid: vec for uid, vec in zip(d_wild_q['ids'], d_wild_q['embedding'])}

        aligned_mut_s = d_mut_s['embedding']
        struct_dim = aligned_mut_s.shape[1]
        seq_dim = d_mut_q['embedding'].shape[1]

        aligned_mut_q, aligned_wild_s, aligned_wild_q = [], [], []
        for uid in anchor_ids:
            aligned_mut_q.append(dict_mut_q.get(uid, torch.zeros(seq_dim)))
            aligned_wild_s.append(dict_wild_s.get(uid, torch.zeros(struct_dim)))
            aligned_wild_q.append(dict_wild_q.get(uid, torch.zeros(seq_dim)))

        prot_data = {
            'mut_s': aligned_mut_s.float(),
            'mut_q': torch.stack(aligned_mut_q).float(),
            'wild_s': torch.stack(aligned_wild_s).float(),
            'wild_q': torch.stack(aligned_wild_q).float()
        }

        return drug_data, prot_data, torch.tensor(item['label'], dtype=torch.float32)


def joint_collate_fn(batch):
    drug_items = [b[0] for b in batch]
    prot_items = [b[1] for b in batch]
    labels = [b[2] for b in batch]

    # --- 药物打包 (PyG 适配版) ---
    brics_seqs = [item['brics_seq'] for item in drug_items]
    brics_padded = pad_sequence(brics_seqs, batch_first=True, padding_value=0)

    # 使用 PyG 的 Batch 工具优雅地组装图批次
    graph_data_list = [item['graph_data'] for item in drug_items]
    graph_batch = Batch.from_data_list(graph_data_list)

    drug_batch = {
        'brics_seq': brics_padded,
        'graph_batch': graph_batch,  # 与 GNN.py 的期待完美契合
        'fingerprint': torch.stack([item['fingerprint'] for item in drug_items])
    }

    # --- 蛋白打包 (保持不变) ---
    mut_s = [item['mut_s'] for item in prot_items]
    mut_q = [item['mut_q'] for item in prot_items]
    wild_s = [item['wild_s'] for item in prot_items]
    wild_q = [item['wild_q'] for item in prot_items]

    lengths = torch.tensor([x.size(0) for x in mut_s])
    mut_s_pad = pad_sequence(mut_s, batch_first=True, padding_value=0.0)
    mut_q_pad = pad_sequence(mut_q, batch_first=True, padding_value=0.0)
    wild_s_pad = pad_sequence(wild_s, batch_first=True, padding_value=0.0)
    wild_q_pad = pad_sequence(wild_q, batch_first=True, padding_value=0.0)

    max_len = mut_s_pad.size(1)
    prot_mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return drug_batch, mut_s_pad, mut_q_pad, wild_s_pad, wild_q_pad, prot_mask, torch.stack(labels)