import pandas as pd
import numpy as np

# 1. 读取数据
print("正在读取数据...")
df = pd.read_csv('relation_with_smiles.csv')

# 检查一下总共有多少个独特的突变
total_unique_mutations = df['Mutation_ID'].nunique()
print(f"总共有 {total_unique_mutations} 种独特的突变 (Mutation_ID)")

# 提取基因名称
df['Gene'] = df['Mutation_ID'].apply(lambda x: str(x).split('_')[0])
unique_genes = df['Gene'].unique()

# 统计每个基因下有多少个【独特的突变】
gene_mut_counts = df.groupby('Gene')['Mutation_ID'].nunique().to_dict()

# 我们的目标：测试集里的独特突变数正好等于 50
target_mutations = 50

best_test_genes = []
found_perfect_split = False
best_test_df = None

print("开始开启智能搜索：寻找独特突变总数恰好为50，且正负比例健康的基因组合...")

# 暴力搜索，最多尝试 50000 次组合
for seed in range(50000):
    np.random.seed(seed)
    # 将基因随机打乱
    shuffled_genes = np.random.permutation(unique_genes)

    test_genes = []
    current_mut_count = 0

    # 尝试把基因逐个放入测试集
    for gene in shuffled_genes:
        muts_in_gene = gene_mut_counts[gene]

        # 如果加上这个基因，突变数超过了50，就跳过这个基因，看下一个能不能塞进去
        if current_mut_count + muts_in_gene > target_mutations:
            continue

        current_mut_count += muts_in_gene
        test_genes.append(gene)

        # 🌟 如果恰好凑齐了50个突变！
        if current_mut_count == target_mutations:
            # 校验测试集的正负样本比例
            test_df = df[df['Gene'].isin(test_genes)]
            pos_ratio = test_df['Relation'].mean()

            # 要求正类样本比例在 35% 到 65% 之间 (保证能画出漂亮的ROC和PR曲线)
            if 0.35 <= pos_ratio <= 0.65:
                best_test_genes = test_genes
                found_perfect_split = True
                best_test_df = test_df
                break  # 找到完美组合，跳出内层循环

    if found_perfect_split:
        break  # 找到完美组合，跳出外层循环

if found_perfect_split:
    # 根据找到的基因组合，正式切分数据
    train_df = df[~df['Gene'].isin(best_test_genes)].copy()
    test_df = df[df['Gene'].isin(best_test_genes)].copy()

    # 获取统计信息
    test_unique_muts = test_df['Mutation_ID'].nunique()
    test_unique_genes = test_df['Gene'].nunique()

    # 删除临时添加的 'Gene' 列，保持与原始文件格式一致
    train_df = train_df.drop(columns=['Gene'])
    test_df = test_df.drop(columns=['Gene'])

    # 2. 保存最终文件
    train_file = 'train_cv_ZeroShot_50mut.csv'
    test_file = 'test_independent_ZeroShot_50mut.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\n✅ 成功找到并划分完毕！")
    print("-" * 40)
    print(f"已生成训练集文件: '{train_file}' ({len(train_df)} 行)")
    print(f"已生成测试集文件: '{test_file}' ({len(test_df)} 行)")
    print("-" * 40)
    print(f"【测试集专属校验】")
    print(f"🎯 包含的独特突变种类 (Mutation_ID): {test_unique_muts} 个")
    print(f"🧬 包含的独立基因: {test_unique_genes} 个 (与训练集 0 重叠！)")
    print(f"⚖️ 正负样本分布:\n{test_df['Relation'].value_counts()}")
else:
    print("\n❌ 未能找到恰好满足条件的组合。")