import torch
import os
import sys
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, Polypeptide
from transformers import AutoTokenizer, EsmModel
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_ROOT, ESM2_PATH

# 忽略 Biopython 可能产生的 PDB 构建警告
warnings.filterwarnings("ignore")


# ===========================================================
# 核心函数 (保持不变)
# ===========================================================

def get_residue_id(residue):
    """
    生成一个唯一的残基标识符 (Chain, ResidueNum, InsertionCode)。
    """
    chain_id = residue.get_parent().id
    res_id = residue.get_id()
    return (chain_id, res_id[1], res_id[2])


def extract_esm2_pocket_embeddings(full_pdb_path, pocket_pdb_path, local_model_dir, tokenizer, model):
    # 1. 解析全长结构 (保持不变)
    parser = PDBParser(QUIET=True)
    full_structure = parser.get_structure("full", full_pdb_path)

    sequence_list = []
    pdb_to_seq_index = {}
    current_seq_index = 0

    for model_struct in full_structure:
        for chain in model_struct:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    res_name_3 = residue.get_resname()
                    res_name_1 = Polypeptide.protein_letters_3to1.get(res_name_3.upper(), 'X')
                    sequence_list.append(res_name_1)
                    unique_id = get_residue_id(residue)
                    pdb_to_seq_index[unique_id] = current_seq_index
                    current_seq_index += 1
        break

    full_sequence = "".join(sequence_list)

    # 2. ESM-2 推理 (保持不变)
    inputs = tokenizer(full_sequence, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)

    # 3. 【关键修改】提取 Pocket 特征并记录 IDs
    pocket_structure = parser.get_structure("pocket", pocket_pdb_path)
    pocket_vectors = []
    pocket_ids = []  # 新增: 用于存储 ID

    for model_struct in pocket_structure:
        for chain in model_struct:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    unique_id = get_residue_id(residue) # (Chain, ResNum, Ins)

                    if unique_id in pdb_to_seq_index:
                        seq_idx = pdb_to_seq_index[unique_id]
                        tensor_idx = seq_idx + 1
                        vector = last_hidden_state[tensor_idx]

                        pocket_vectors.append(vector.cpu())
                        pocket_ids.append(unique_id) # 同步保存 ID
                    else:
                        pass
        break

    if not pocket_vectors:
        return None

    # 【关键修改】返回字典而不是 Tensor
    return {
        "embedding": torch.stack(pocket_vectors),
        "ids": pocket_ids
    }


# ===========================================================
# 主程序修改版：针对 WT 口袋的反向匹配逻辑
# ===========================================================
if __name__ == "__main__":
    # --- 1. 路径配置 (修改为 Wild 路径) ---

    # 全长 WT 所在文件夹 (例如 F5H094.pdb 所在位置)
    FULL_WT_DIR = str(FEATURE_ROOT / "wild")

    # WT 口袋所在文件夹 (例如 F5H094_M233I_WT_pocket.pdb 所在位置)
    WT_POCKET_DIR = str(FEATURE_ROOT / "wild" / "pockets_extracted")

    # 结果保存位置
    OUTPUT_DIR = str(FEATURE_ROOT / "wild" / "esm2_features")

    # 本地模型路径
    if not ESM2_PATH:
        raise SystemExit("Please set GEOCDL_ESM2_PATH to a local ESM-2 checkpoint directory.")
    LOCAL_ESM_PATH = ESM2_PATH

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 2. 加载模型 ---
    print(f"正在加载 ESM-2 模型: {LOCAL_ESM_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_ESM_PATH, local_files_only=True)
        model = EsmModel.from_pretrained(LOCAL_ESM_PATH, local_files_only=True)
        model = model.cuda()  # 建议使用 GPU
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # --- 3. 遍历逻辑：以 WT Pocket 为基准 ---

    # 获取所有的 WT 口袋文件
    # 文件名示例: F5H094_M233I_WT_pocket.pdb
    pocket_files = [f for f in os.listdir(WT_POCKET_DIR) if f.endswith("_WT_pocket.pdb")]
    total_files = len(pocket_files)

    print(f"共发现 {total_files} 个 WT 口袋文件待处理。")

    success_count = 0
    fail_count = 0

    for i, pocket_filename in enumerate(pocket_files):
        # 1. 解析文件名，获取 PDB ID
        # pocket_filename: "F5H094_M233I_WT_pocket.pdb"
        # split('_')[0] -> "F5H094"
        pdb_id = pocket_filename.split('_')[0]

        # 2. 构建对应的全长 WT 文件路径
        # 假设全长文件名为 "F5H094.pdb"
        full_wt_filename = f"{pdb_id}.pdb"
        full_wt_path = os.path.join(FULL_WT_DIR, full_wt_filename)

        pocket_path = os.path.join(WT_POCKET_DIR, pocket_filename)

        # 3. 检查全长文件是否存在
        if not os.path.exists(full_wt_path):
            print(
                f"[{i + 1}/{total_files}] 跳过: 找不到对应的全长 WT 文件 -> {full_wt_filename} (对应 {pocket_filename})")
            fail_count += 1
            continue

        try:
            print(f"[{i + 1}/{total_files}] 处理: {pocket_filename} <- 源文件: {full_wt_filename} ...", end="\r")

            # 4. 提取特征
            embedding = extract_esm2_pocket_embeddings(
                full_wt_path,  # 输入全长 WT
                pocket_path,  # 输入 WT 口袋 (用于 Mask)
                LOCAL_ESM_PATH,
                tokenizer,
                model
            )

            if embedding is not None:
                # 既然这是 ESM2_wild.py，我们确定是 wild 逻辑
                save_name = pocket_filename.replace("_pocket.pdb", ".pt")
                save_path = os.path.join(OUTPUT_DIR, save_name)

                # 直接保存包含 'embedding' 和 'ids' 的字典
                torch.save(embedding, save_path)
                success_count += 1
            else:
                print(f"\n[{i + 1}/{total_files}] 失败: 特征提取为空 (可能口袋残基未匹配到全长序列)")
                fail_count += 1

        except Exception as e:
            print(f"\n[{i + 1}/{total_files}] 错误 {pocket_filename}: {e}")
            fail_count += 1

    print("\n" + "=" * 30)
    print("WT 特征提取完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"输出目录: {OUTPUT_DIR}")
