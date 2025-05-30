import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, levene, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import pingouin as pg
import scikit_posthocs as sp
import itertools
import os
from collections import defaultdict
from itertools import combinations

src_dir = os.path.dirname(os.path.abspath(__file__))


def assign_letters(models, pval_matrix, data, metric):
    """根据模型性能中位数排序并分配显著性字母标签（支持多个标签）"""
    from collections import defaultdict

    # 按中位数性能从高到低排序模型
    medians = {model: np.median(data[data['Model'] == model][metric]) for model in models}
    sorted_models = sorted(models, key=lambda x: -medians[x])

    # 构建显著性差异对
    sig_pairs = defaultdict(set)
    for (model1, model2) in combinations(sorted_models, 2):
        p = pval_matrix.loc[model1, model2]
        if p < 0.05:
            sig_pairs[model1].add(model2)
            sig_pairs[model2].add(model1)

    # 构建非显著性分组（cliques）
    groups = []
    for model in sorted_models:
        added = False
        for group in groups:
            # 当前模型与组内所有模型均无显著差异才加入
            if all(m not in sig_pairs[model] for m in group):
                group.append(model)
                added = True
        if not added:
            groups.append([model])

    # 分配字母，每个组一个字母，模型可能被分配多个字母
    letter_map = defaultdict(str)
    for i, group in enumerate(groups):
        letter = chr(97 + i)  # 'a', 'b', 'c', ...
        for model in group:
            if letter not in letter_map[model]:
                letter_map[model] += letter

    return letter_map

def perform_test(data, metric):
    """执行统计检验并返回字母标记"""
    models = data['Model'].unique()
    

    # Kruskal + Dunn (非参数)
    pvals = sp.posthoc_dunn(data, val_col=metric, group_col='Model', p_adjust='bonferroni')

    # 确保矩阵对称
    models_sorted = sorted(models)
    pvals = pvals.reindex(index=models_sorted, columns=models_sorted)
    # print(pvals)
    np.fill_diagonal(pvals.values, 1.0)

    return assign_letters(models_sorted, pvals, data, metric)


def analyze_and_save_results():
    """主分析函数"""
    # 读取数据
    input_path = f'{src_dir}/../results/auc_tss.csv'
    df = pd.read_csv(input_path)
    # 列名标准化
    column_map = {
        'model': 'Model',
        'auc': 'AUC',
        'tss': 'TSS',
        'class': 'class'
    }
    df = df.rename(columns={k:v for k,v in column_map.items() if k in df.columns})
    
    # 初始化结果存储
    all_results = []
    required_models = ['GLM', 'GAM', 'SVM', 'MaxEnt', 'RF']

    # 遍历每个class和指标
    for (class_name, group_df), metric in itertools.product(df.groupby('class'), ['AUC', 'TSS']):
        print(f"\n正在分析 {class_name} 类别的 {metric} 指标...")

        letter_map = perform_test(group_df, metric)
        
        for model, letter in letter_map.items():
            all_results.append({
                'class': class_name,
                'metric': metric,
                'model': model,
                'significance': letter
            })

    # 转换为DataFrame
    result_df = pd.DataFrame(all_results)
    
    # 转换为宽格式
    try:
        pivot_df = result_df.pivot_table(
            index=['class', 'model'], 
            columns='metric', 
            values='significance',
            aggfunc='first'
        ).reset_index()
        pivot_df.columns.name = None
        pivot_df = pivot_df[['class', 'model', 'AUC', 'TSS']]
    except KeyError as e:
        print("转换宽表时出错，当前数据:")
        print(result_df)
        raise

    # 保存结果
    output_file = f'{src_dir}/../results/statistical_significance_results.csv'
    pivot_df.to_csv(output_file, index=False)


# 运行分析
if __name__ == "__main__":
    analyze_and_save_results()