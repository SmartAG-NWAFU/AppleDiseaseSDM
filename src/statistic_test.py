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
    """Assign significance letter labels based on model performance median ranking (supports multiple labels)"""
    from collections import defaultdict

    # Sort models by median performance from high to low
    medians = {model: np.median(data[data['Model'] == model][metric]) for model in models}
    sorted_models = sorted(models, key=lambda x: -medians[x])

    # Build significance difference pairs
    sig_pairs = defaultdict(set)
    for (model1, model2) in combinations(sorted_models, 2):
        p = pval_matrix.loc[model1, model2]
        if p < 0.05:
            sig_pairs[model1].add(model2)
            sig_pairs[model2].add(model1)

    # Build non-significant groups (cliques)
    groups = []
    for model in sorted_models:
        added = False
        for group in groups:
            # Add current model to group only if it has no significant difference with all models in the group
            if all(m not in sig_pairs[model] for m in group):
                group.append(model)
                added = True
        if not added:
            groups.append([model])

    # Assign letters, one letter per group, models may be assigned multiple letters
    letter_map = defaultdict(str)
    for i, group in enumerate(groups):
        letter = chr(97 + i)  # 'a', 'b', 'c', ...
        for model in group:
            if letter not in letter_map[model]:
                letter_map[model] += letter

    return letter_map

def perform_test(data, metric):
    """Perform statistical test and return letter labels"""
    models = data['Model'].unique()
    

    # Kruskal + Dunn (non-parametric)
    pvals = sp.posthoc_dunn(data, val_col=metric, group_col='Model', p_adjust='bonferroni')

    # Ensure matrix symmetry
    models_sorted = sorted(models)
    pvals = pvals.reindex(index=models_sorted, columns=models_sorted)
    # print(pvals)
    np.fill_diagonal(pvals.values, 1.0)

    return assign_letters(models_sorted, pvals, data, metric)


def analyze_and_save_results():
    """Main analysis function"""
    # Read data
    input_path = f'{src_dir}/../results/auc_tss.csv'
    df = pd.read_csv(input_path)
    # Standardize column names
    column_map = {
        'model': 'Model',
        'auc': 'AUC',
        'tss': 'TSS',
        'class': 'class'
    }
    df = df.rename(columns={k:v for k,v in column_map.items() if k in df.columns})
    
    # Initialize result storage
    all_results = []
    required_models = ['GLM', 'GAM', 'SVM', 'MaxEnt', 'RF']

    # Iterate through each class and metric
    for (class_name, group_df), metric in itertools.product(df.groupby('class'), ['AUC', 'TSS']):
        print(f"\nAnalyzing {metric} metric for {class_name} class...")

        letter_map = perform_test(group_df, metric)
        
        for model, letter in letter_map.items():
            all_results.append({
                'class': class_name,
                'metric': metric,
                'model': model,
                'significance': letter
            })

    # Convert to DataFrame
    result_df = pd.DataFrame(all_results)
    
    # Convert to wide format
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
        print("Error converting to wide format, current data:")
        print(result_df)
        raise

    # Save results
    output_file = f'{src_dir}/../results/statistical_significance_results.csv'
    pivot_df.to_csv(output_file, index=False)


# 运行分析
if __name__ == "__main__":
    analyze_and_save_results()