import os
import re
import glob
import sys
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_ROOT


# ==========================================
# 核心类与函数
# ==========================================

class PocketSelect(Select):
    """
    自定义选择器：仅保留 selected_residues 集合中的残基。
    """

    def __init__(self, selected_residues):
        self.selected_residues = selected_residues

    def accept_residue(self, residue):
        return residue in self.selected_residues


def extract_pocket(structure, output_path, chain_id, res_id, radius=10.0):
    """
    【函数说明】从突变体PDB结构中提取指定位点的口袋区域（核心功能）
    参数：
        structure：Bio.PDB解析后的突变体结构对象
        output_path：口袋PDB文件保存路径
        chain_id：目标残基所在链ID（默认A链）
        res_id：目标残基编号（如233）
        radius：口袋提取半径（10Å）
    返回：bool，成功返回True，失败返回False
    """
    model = structure[0]  # 取第一个模型

    # 1. 获取目标链 (Chain)
    try:
        chain = model[chain_id]
    except KeyError:

        first_chain = list(model.get_chains())[0]
        print(f"警告: 链 '{chain_id}' 不存在，自动切换为链 '{first_chain.id}'")
        chain = first_chain
        chain_id = chain.id

    # 2. 定位锚点残基
    # Bio.PDB ID 格式: (' ', res_id, ' ')
    target_res_key = (' ', res_id, ' ')

    try:
        target_res = chain[target_res_key]
    except KeyError:
        print(f"错误: 无法在链 {chain_id} 中找到残基 {res_id}。跳过此文件。")
        return False

    # 3. 计算几何中心 (Centroid)
    # 提取目标残基所有原子的3维坐标
    atom_coords = [atom.get_coord() for atom in target_res.get_atoms()]
    if not atom_coords:
        return False
    centroid = np.mean(atom_coords, axis=0)

    # 4. 邻居搜索 (Neighbor Search)
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)
    atoms_nearby = ns.search(centroid, radius, level='A')

    # 5. 收集残基 (Selection)
    selected_residues = set()
    selected_residues.add(target_res)  # 必选中心

    for atom in atoms_nearby:
        parent_res = atom.get_parent()
        # 排除水 (HOH) 等非标准残基，仅保留标准氨基酸 (id[0] == ' ')
        if parent_res.id[0] == ' ':
            selected_residues.add(parent_res)

    # 6. 保存结果
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(output_path, PocketSelect(selected_residues))
        return True
    except Exception as e:
        print(f"保存出错: {e}")
        return False


# ==========================================
# 批量处理逻辑 (针对你的文件名格式)
# ==========================================

def parse_mutation_info(filename):
    """
       【函数说明】从突变体文件名解析残基编号
       参数：filename - 突变体文件名（如F5H094_M233I.pdb）
       返回：残基编号（如233），解析失败返回None
       """
    # 1. 去掉扩展名 -> F5H094_M233I
    base_name = os.path.splitext(filename)[0]

    # 2. 获取最后一个下划线后的部分 -> M233I
    # split('_')[-1] 会取最后一部分
    mutation_str = base_name.split('_')[-1]

    # 3. 使用正则表达式提取数字
    # 匹配规则: 一个字母 + (数字) + 一个字母
    # 例如 M233I -> 匹配 233
    match = re.search(r'[A-Za-z](\d+)[A-Za-z]', mutation_str)

    if match:
        res_num = int(match.group(1))
        return res_num
    else:
        return None


def main():
    # 1. 设置输入输出路径
    input_dir = str(FEATURE_ROOT / "mut")
    output_dir = os.path.join(input_dir, "pockets_extracted")  # 结果保存在子文件夹

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 2. 获取所有 pdb 文件
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))

    print(f"在 {input_dir} 中发现 {len(pdb_files)} 个 PDB 文件。\n")

    parser = PDBParser(QUIET=True)
    count = 0

    for pdb_path in pdb_files:
        filename = os.path.basename(pdb_path)

        # 解析残基号
        res_num = parse_mutation_info(filename)

        if res_num is None:
            print(f"[跳过] 无法从文件名解析突变位点: {filename}")
            continue

        # 设置输出路径
        # 输出文件名示例: F5H094_M233I_pocket.pdb
        out_name = filename.replace(".pdb", "_pocket.pdb")
        out_path = os.path.join(output_dir, out_name)

        # 默认链 ID (FoldX 输出通常为 A，代码内部有容错机制会自动切到第一条链)
        target_chain = 'A'

        # 解析结构
        try:
            structure = parser.get_structure(filename, pdb_path)

            # 执行提取
            success = extract_pocket(
                structure=structure,
                output_path=out_path,
                chain_id=target_chain,
                res_id=res_num,
                radius=10.0
            )

            if success:
                print(f"[成功] {filename} -> 提取中心: {res_num} -> 输出: {out_name}")
                count += 1

        except Exception as e:
            print(f"[错误] 处理 {filename} 时发生异常: {e}")

    print(f"\n全部完成! 成功提取 {count} 个口袋结构。")


if __name__ == "__main__":
    main()
