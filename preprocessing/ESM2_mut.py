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


def get_residue_id(residue):
    """
    生成一个唯一的残基标识符 (Chain, ResidueNum, InsertionCode)。
    Biopython 的 get_id() 返回 (HeteroFlag, SequenceNum, InsertCode)。
    我们需要结合 Chain ID 确保唯一性。
    """
    chain_id = residue.get_parent().id
    # res_id[1] 是序列号, res_id[2] 是插入码 (通常为空格)
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
if __name__ == "__main__":
    # --- 1. 路径配置 (注意使用 r"" 来防止转义字符问题) ---

    # 包含所有原始 PDB 的文件夹
    full_pdb_dir = str(FEATURE_ROOT / "mut")

    # 包含所有口袋 PDB 的文件夹
    pocket_pdb_dir = str(FEATURE_ROOT / "mut" / "pockets_extracted")

    # 模型文件夹
    if not ESM2_PATH:
        raise SystemExit("Please set GEOCDL_ESM2_PATH to a local ESM-2 checkpoint directory.")
    local_esm_path = ESM2_PATH

    # 结果保存文件夹 (代码会自动创建)
    output_dir = str(FEATURE_ROOT / "mut" / "old" / "esm2_features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. 加载模型 (只加载一次，节省时间) ---
    print(f"正在加载模型: {local_esm_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_esm_path, local_files_only=True)
        model = EsmModel.from_pretrained(local_esm_path, local_files_only=True)
        # 如果有显卡，取消下面这行的注释
        model = model.cuda()
        model.eval()
        print("模型加载成功！开始批量处理...")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # --- 3. 遍历并处理 ---
    # 获取所有全长 PDB 文件
    files = [f for f in os.listdir(full_pdb_dir) if f.endswith(".pdb")]
    total_files = len(files)

    print(f"共发现 {total_files} 个 PDB 文件待处理。")

    success_count = 0
    fail_count = 0

    for i, filename in enumerate(files):
        # 构建全长 PDB 的完整路径
        full_path = os.path.join(full_pdb_dir, filename)

        # 构建口袋 PDB 的完整路径
        # 【注意】这里假设口袋文件和全长文件同名。
        # 如果口袋文件名是 "1abc_pocket.pdb"，请修改下面这行为：
        pocket_filename = filename.replace(".pdb", "_pocket.pdb")
        pocket_path = os.path.join(pocket_pdb_dir, pocket_filename)

        # 检查对应的口袋文件是否存在
        if not os.path.exists(pocket_path):
            print(f"[{i + 1}/{total_files}] 跳过: 找不到对应的口袋文件 -> {pocket_filename}")
            fail_count += 1
            continue

        try:
            print(f"[{i + 1}/{total_files}] 正在处理: {filename} ...", end="\r")

            # 调用提取函数
            embedding = extract_esm2_pocket_embeddings(
                full_path,
                pocket_path,
                local_esm_path,
                tokenizer,
                model
            )

            if embedding is not None:
                # 保持文件名逻辑不变
                # Mut: filename.replace(".pdb", ".pt")
                # Wild: pocket_filename.replace("_pocket.pdb", ".pt")
                if "mut" in output_dir:
                    save_name = filename.replace(".pdb", ".pt")
                else:
                    save_name = pocket_filename.replace("_pocket.pdb", ".pt")

                save_path = os.path.join(output_dir, save_name)

                # 直接保存包含 'embedding' 和 'ids' 的字典
                torch.save(embedding, save_path)
                success_count += 1
            else:
                print(f"\n[{i + 1}/{total_files}] 失败: 无法提取特征 {filename}")
                fail_count += 1

        except Exception as e:
            print(f"\n[{i + 1}/{total_files}] 错误 {filename}: {e}")
            fail_count += 1

    print("\n" + "=" * 30)
    print("处理完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"特征文件已保存至: {output_dir}")
