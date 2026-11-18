import os
import glob
import sys
import torch
import numpy as np
from pathlib import Path
import biotite.structure as struc
import biotite.structure.io as strucio
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_ROOT, GVP_WEIGHTS

# 引入你的 GVP 库
import gvp.models
import gvp.data

# ================= 配置区域 =================

# 1. 路径设置。通过 GEOCDL_PROTEIN_TYPE=mut|wild 选择要处理的蛋白类型。
PROTEIN_TYPE = os.environ.get("GEOCDL_PROTEIN_TYPE", "wild")
if PROTEIN_TYPE == "mut":
    POCKET_DIR = str(FEATURE_ROOT / "mut" / "pockets_extracted")
    OUTPUT_DIR = str(FEATURE_ROOT / "mut" / "old" / "pocket_embeddings1")
else:
    POCKET_DIR = str(FEATURE_ROOT / "wild" / "pockets_extracted")
    OUTPUT_DIR = str(FEATURE_ROOT / "wild" / "pocket_embeddings1")
MODEL_PATH = GVP_WEIGHTS

# 2. 模型参数 (严格匹配训练时的配置)
NODE_DIM = (512, 16)  # 隐藏层 (scalar, vector)
EDGE_DIM = (32, 1)  # 边隐藏层
NUM_LAYERS = 3  # 层数
MODEL_INPUT_DIM = (6, 3)

# 输入边维度设为 32
INPUT_EDGE_DIM = (32, 1)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 核心工具函数 =================

def get_backbone_features(coords_n, coords_ca, coords_c, coords_o, dihedrals):
    """
    计算 GVP 标准的 (6, 3) 输入特征。
    Vector: 归一化的骨架方向向量 (3个)
    Scalar: 主链二面角 phi, psi, omega 的 sin 和 cos 值 (共6维)
    """

    def norm_vec(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n != 0)

    # 1. 计算方向向量 (Vector Features)
    vec_n = norm_vec(coords_n - coords_ca)
    vec_c = norm_vec(coords_c - coords_ca)
    vec_o = norm_vec(coords_o - coords_ca)

    # [N, 3, 3]
    node_v = np.stack([vec_n, vec_c, vec_o], axis=1)

    # 2. 计算标量特征 (Scalar Features)
    node_s = []
    for (phi, psi, omega) in dihedrals:

        phi = 0.0 if np.isnan(phi) else phi
        psi = 0.0 if np.isnan(psi) else psi
        omega = 0.0 if np.isnan(omega) else omega

        # 展开为 6 维: [sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)]
        node_s.append([
            np.sin(phi), np.cos(phi),
            np.sin(psi), np.cos(psi),
            np.sin(omega), np.cos(omega)
        ])

    node_s = np.array(node_s, dtype=np.float32)

    return torch.tensor(node_s, dtype=torch.float32), torch.tensor(node_v, dtype=torch.float32)


def process_pdb_to_graph(pdb_path, top_k=30):
    try:
        # 加载结构
        array = strucio.load_structure(pdb_path)
        array = array[struc.filter_amino_acids(array)]

        # 【新增逻辑】利用 Biotite 计算整个结构的骨架二面角
        # 返回的是与残基数量长度相同的数组 (phi, psi, omega)
        phi_arr, psi_arr, omega_arr = struc.dihedral_backbone(array)
        residue_ids, _ = struc.get_residues(array)

        n_coords, ca_coords, c_coords, o_coords = [], [], [], []
        recorded_ids = []
        valid_dihedrals = []

        for i, res_id in enumerate(residue_ids):
            res_mask = array.res_id == res_id
            res_atoms = array[res_mask]
            try:
                # 获取原子坐标
                n = res_atoms[res_atoms.atom_name == "N"][0].coord
                ca = res_atoms[res_atoms.atom_name == "CA"][0].coord
                c = res_atoms[res_atoms.atom_name == "C"][0].coord
                o_candidates = res_atoms[np.isin(res_atoms.atom_name, ["O", "OXT"])]
                o = o_candidates[0].coord if len(o_candidates) > 0 else c

                n_coords.append(n)
                ca_coords.append(ca)
                c_coords.append(c)
                o_coords.append(o)

                # 提取 ID 以匹配 Biopython 格式
                curr_chain = res_atoms.chain_id[0]
                curr_num = int(res_atoms.res_id[0])
                curr_ins = res_atoms.ins_code[0]
                if curr_ins == "": curr_ins = " "

                recorded_ids.append((curr_chain, curr_num, curr_ins))

                # 【新增】记录当前有效残基对应的二面角
                valid_dihedrals.append((phi_arr[i], psi_arr[i], omega_arr[i]))

            except IndexError:
                continue

        if len(ca_coords) == 0: return None, None

        n_coords = np.array(n_coords)
        ca_coords = np.array(ca_coords)
        c_coords = np.array(c_coords)
        o_coords = np.array(o_coords)

        # 1. 节点特征 (Node Features) - 传入二面角
        node_s, node_v = get_backbone_features(n_coords, ca_coords, c_coords, o_coords, valid_dihedrals)

        # 2. 构建边 (Edges - KNN)
        dist_matrix = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        num_nodes = len(ca_coords)
        src_list, dst_list = [], []

        k = min(top_k, num_nodes - 1)
        for i in range(num_nodes):
            dists = dist_matrix[i]
            sorted_indices = np.argsort(dists)
            neighbors = sorted_indices[1:k + 1]  # 排除自己
            for n_idx in neighbors:
                if dists[n_idx] < 20.0:  # 截断距离
                    src_list.append(i)
                    dst_list.append(n_idx)

        if len(src_list) == 0: return None, None
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # 3. 边特征 (Edge Features)
        dataset_idx = torch.tensor(src_list)
        neighbor_idx = torch.tensor(dst_list)

        diff = torch.tensor(ca_coords[neighbor_idx] - ca_coords[dataset_idx], dtype=torch.float32)
        dist = torch.norm(diff, dim=-1, keepdim=True)

        # Edge Vector
        edge_v = diff / (dist + 1e-6)
        edge_v = edge_v.unsqueeze(1)  # [E, 1, 3]

        # Edge Scalar: RBF 编码
        rbf_centers = torch.linspace(0, 20, 16)
        rbf_sigma = (20.0 / 16)
        edge_rbf = torch.exp(-((dist - rbf_centers) ** 2) / (rbf_sigma ** 2))

        # Padding: 补零到 32 维
        padding = torch.zeros(edge_rbf.shape[0], 16)
        edge_s = torch.cat([edge_rbf, padding], dim=-1)  # [E, 32]

        graph_data = Data(
            x=node_s,
            node_v=node_v,
            edge_index=edge_index,
            edge_s=edge_s,
            edge_v=edge_v,
            mask=torch.ones(num_nodes, dtype=torch.bool)
        )

        return graph_data, recorded_ids

    except Exception as e:
        print(f"Skipping {pdb_path}: {e}")
        return None, None


# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading model from {MODEL_PATH}...")

    # 初始化模型
    model = gvp.models.CPDModel(
        node_in_dim=MODEL_INPUT_DIM,
        node_h_dim=NODE_DIM,
        edge_in_dim=INPUT_EDGE_DIM,
        edge_h_dim=EDGE_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    # 加载权重
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    pdb_files = glob.glob(os.path.join(POCKET_DIR, "*.pdb"))
    print(f"Found {len(pdb_files)} pdb files.")

    for pdb_file in tqdm(pdb_files):
        data, ids = process_pdb_to_graph(pdb_file)
        if data is None: continue

        batch = Batch.from_data_list([data]).to(device)

        with torch.no_grad():
            h_V = (batch.x, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)

            h_V = model.W_v(h_V)
            h_E = model.W_e(h_E)

            for layer in model.encoder_layers:
                h_V = layer(h_V, batch.edge_index, h_E)

            final_emb = h_V[0]  # 取出标量特征 [N, 512]

        base_name = os.path.basename(pdb_file).replace(".pdb", ".pt")
        save_path = os.path.join(OUTPUT_DIR, base_name)

        torch.save({
            "embedding": final_emb.cpu(),
            "ids": ids
        }, save_path)

    print(f"\nAll Done! Embeddings saved to {OUTPUT_DIR}")
