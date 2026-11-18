import os
import re
import glob
import sys
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_ROOT

# ================= 配置区域 =================
# 突变体 PDB 文件夹 (用于获取文件名和位点信息)
MUT_DIR = str(FEATURE_ROOT / "mut")

# 野生型 全长 PDB 文件夹 (源文件)
WILD_FULL_DIR = str(FEATURE_ROOT / "wild")

# 输出：野生型 口袋 PDB 文件夹
OUTPUT_WT_POCKET_DIR = str(FEATURE_ROOT / "wild" / "pockets_extracted")

if not os.path.exists(OUTPUT_WT_POCKET_DIR):
    os.makedirs(OUTPUT_WT_POCKET_DIR)


# ================= 核心类 ================
class PocketSelect(Select):
    def __init__(self, selected_residues):
        self.selected_residues = selected_residues

    def accept_residue(self, residue):
        return residue in self.selected_residues


def extract_pocket(structure, output_path, res_id, radius=10.0):
    """
    【函数说明】从PDB结构中提取指定位点的口袋区域（核心功能）
    参数：
        structure：Bio.PDB解析后的结构对象（包含蛋白的原子/残基/链信息）
        output_path：口袋PDB文件的保存路径
        res_id：目标残基的编号（如233）
        radius：口袋提取半径（默认10Å，Å是原子距离单位）
    返回：bool，提取成功返回True，失败返回False
    """
    model = structure[0]
    # 尝试自动寻找包含该位点的链
    target_res = None
    chain_id = None

    # 遍历所有链寻找该位点
    for chain in model:
        if (' ', res_id, ' ') in chain:
            target_res = chain[(' ', res_id, ' ')]
            chain_id = chain.id
            break

    if target_res is None:
        print(f"  [Error] 无法在结构中找到位点 {res_id}")
        return False

    # 计算中心
    atom_coords = [atom.get_coord() for atom in target_res.get_atoms()]
    if not atom_coords: return False
    centroid = np.mean(atom_coords, axis=0)

    # 邻居搜索
    atoms = list(model.get_atoms())
    ns = NeighborSearch(atoms)
    atoms_nearby = ns.search(centroid, radius, level='R')  # level='R' 直接返回残基

    # 筛选标准氨基酸
    selected_residues = set()
    for res in atoms_nearby:
        if res.id[0] == ' ':
            selected_residues.add(res)

    # 保存
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(output_path, PocketSelect(selected_residues))
        return True
    except Exception as e:
        print(f"  [Save Error] {e}")
        return False


# ================= 主程序 =================
def main():
    # 1. 获取突变体文件列表 (作为索引)
    mut_files = glob.glob(os.path.join(MUT_DIR, "*.pdb"))
    print(f"扫描到 {len(mut_files)} 个突变体文件作为索引参考。")

    parser = PDBParser(QUIET=True)
    count = 0

    for mut_path in mut_files:
        filename = os.path.basename(mut_path)  # e.g., F5H094_M233I.pdb

        # --- 解析信息 ---
        base_name = os.path.splitext(filename)[0]  # F5H094_M233I
        parts = base_name.split('_')

        # 假设命名格式为: ID_Mutation (例如 F5H094_M233I)
        # 此时 ID = parts[0], Mutation = parts[-1]
        pdb_id = parts[0]
        mutation_str = parts[-1]

        # 提取位点数字
        match = re.search(r'(\d+)', mutation_str)
        if not match:
            print(f"跳过: 无法解析突变位点 {filename}")
            continue
        res_num = int(match.group(1))

        # --- 定位 WT 文件 ---
        # 假设 WT 文件名就是 PDB ID (例如 F5H094.pdb)
        # 如果你的 WT 文件名有其他后缀，请在这里修改
        wt_filename = f"{pdb_id}.pdb"
        wt_full_path = os.path.join(WILD_FULL_DIR, wt_filename)

        if not os.path.exists(wt_full_path):
            print(f"跳过: 找不到对应的 WT 全长文件 -> {wt_filename}")
            continue

        # --- 设置输出路径 ---
        # 命名为 F5H094_M233I_WT_pocket.pdb
        # 这样文件名里包含了它对应的突变信息，方便后续 Dataset 配对
        save_name = f"{base_name}_WT_pocket.pdb"
        save_path = os.path.join(OUTPUT_WT_POCKET_DIR, save_name)

        if os.path.exists(save_path):
            # print(f"已存在: {save_name}")
            continue

        print(f"正在提取 WT 口袋: {wt_filename} (位点 {res_num}) -> {save_name} ...")

        try:
            structure = parser.get_structure("WT", wt_full_path)
            success = extract_pocket(structure, save_path, res_num)
            if success:
                count += 1
        except Exception as e:
            print(f"  处理异常: {e}")

    print(f"\n处理完成！共提取 {count} 个 WT 口袋。")
    print(f"保存在: {OUTPUT_WT_POCKET_DIR}")


if __name__ == "__main__":
    main()
