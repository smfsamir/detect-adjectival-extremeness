import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from itertools import combinations
from statsmodels.formula.api import ols
from sklearn.metrics import auc

from ..pkl_operations.pkl_io import store_pic_dynamic

def pairplot_intf_feats(frame, feats, fname):
    sns.pairplot(data=frame, vars=feats, hue='type')
    store_pic_dynamic(plt, fname, 'data/artifacts')

    corrs = np.zeros((len(feats), len(feats)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            corrs[i][j] = spearmanr(frame[feats[i]], frame[feats[j]])[0]
    visualize_correlation_heatmap(corrs, fname, False)

def visualize_correlation_heatmap(correlation_df, fig_name=None, save_as_eps=False):
    fig, axes = plt.subplots(1,1)
    fig.tight_layout()
    fig = sns.heatmap(correlation_df, annot=True, ax=axes, square=True)
    bottom, top = axes.get_ylim()
    left, right = axes.get_xlim()
    axes.set_ylim(bottom + 0.5, top - 0.5)
    print(axes.get_xlim())
    print(axes.get_ylim())
    # axes.set_xlim(left + 0.5, right - 0.5)
    if fig_name:
        if save_as_eps:
            plt.savefig(f"{fig_name}.eps",  format='eps')
        else:
            store_pic_dynamic(plt, f'{fig_name}_heatmap', 'data/artifacts')
    
def pr_curve(precisions, recalls, model_name, fname, title):
    fig, axes = plt.subplots(1,1)
    auc_output = auc(recalls, precisions)
    plt.plot(recalls, precisions, label=f'{model_name} (AUC = {auc_output:.2f})', c='firebrick', linewidth=2)
    plt.title(title)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend()

    axes.autoscale_view()
    plt.ylabel("Precision", fontsize="large")
    plt.xlabel("Recall", fontsize='large')
    plt.title(title)
    print(plt.xlim()[1] - plt.xlim()[0])
    plt.legend(fontsize='large')
    plt.legend(loc='lower center')
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()

    store_pic_dynamic(plt, fname, 'results', False)

def prettify_graph(ax):
    ax.autoscale_view()
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()

def pr_curve_multiple(precisionss, recallss, model_names, title, fname):
    fig, axes = plt.subplots(1,1)
    styles = list(reversed([":",  "--", "-"]))
    cs = list(['firebrick', 'royalblue','darkgreen'])
    fig, ax = plt.subplots()
    for i in range(len(model_names)):
        model_name = model_names[i]
        precisions, recalls = precisionss[i], recallss[i]
        auc_output = auc(recalls, precisions)
        plt.plot(recalls, precisions, label=f"{model_name} (AUC = {auc_output:.2f})", linestyle=styles[i], c=cs[i], linewidth=2)

    ax.autoscale_view()
    plt.ylabel("Precision", fontsize="large")
    plt.xlabel("Recall", fontsize='large')
    plt.title(title)
    print(plt.xlim()[1] - plt.xlim()[0])
    plt.legend(fontsize='large')
    plt.legend(loc='lower center')
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()

    store_pic_dynamic(plt, fname, 'results', False)

def visualize_embed_2d(embeds_one, embeds_two, title):
    def sc_plot_vecs(embeds, c ):
        x, y = zip(*embeds)
        axes.scatter(x, y, c=c)

    fig, axes = plt.subplots(1,1)
    sc_plot_vecs(embeds_one, c='b')
    sc_plot_vecs(embeds_two, c='r')
    plt.savefig(f'data/artifacts/pictures/{title}.png')

def visualize_all_query_exemplar(final_frame, title, fname, xlabel, ylabel, measurement, legend_measure_name):
    fig, ax = plt.subplots(1,1)

    rename_column  = lambda frame, curr_col_name, new_col_name: frame.rename(columns={curr_col_name: f"{new_col_name}"})

    intf_frame = rename_column(final_frame[final_frame['type'] == 'intf'], measurement, f'intensifier {legend_measure_name}')
    ctrl_frame = rename_column(final_frame[final_frame['type'] == 'ctrl'], measurement, f'control {legend_measure_name}')

    sns.kdeplot(intf_frame[f'intensifier {legend_measure_name}'], ax=ax)
    sns.kdeplot(ctrl_frame[f'control {legend_measure_name}'], ax=ax)

    results = mannwhitneyu(intf_frame[f'intensifier {legend_measure_name}'], ctrl_frame[f'control {legend_measure_name}'])
    print(f"Mann-whitney U test results: {results}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"data/artifacts/pictures/{fname}.png")

def plot_loco_results(centroid_aucs, dist_test_aucs):
    fig, ax = plt.subplots(1,1)

    styles = list(reversed([":",  "--", "-"]))
    cs = list(['firebrick', 'royalblue','darkgreen'])
    # change labels to 2 to 10
    ax.plot(np.arange(len(centroid_aucs)), centroid_aucs, marker='o',  linestyle=styles[0], linewidth=2, c=cs[0], label='DSM Centroid')
    ax.plot(np.arange(len(dist_test_aucs)), dist_test_aucs, marker='o', linestyle=styles[1], linewidth=2, c=cs[1], label='Distrib. test')

    # ax.set_xticklabels(np.arange(2, len(centroid_aucs) + 1))

    ax.set_xlabel("Number of clusters (k)", fontsize='large')
    ax.set_ylabel("AUC", fontsize='large')
    plt.legend(fontsize='large')
    plt.legend(loc='lower center')
    plt.xticks(np.arange(11), ['2', '', '4', '', '6', '', '8', '', '10', '', '12'])
    plt.title("Performance for leave-one-cluster-out classification")
    prettify_graph(ax)
    store_pic_dynamic(plt, 'loco_aucs', 'results', True)