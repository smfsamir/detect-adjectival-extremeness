"""This module corresponds to Section 4.3 of the paper. 
"""

import argparse
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from functools import reduce

# from main_ngram_control import conv_match_to_long_form

from packages.pkl_operations.pkl_io import load_pkl_from_path, store_pic_dynamic, store_results_dynamic
from packages.adverbs.constants import *
from packages.utils.util_functions import get_artifacts_path, get_today_date_formatted
from packages.utils.util_functions import map_list
from packages.adverbs.visualizations import pairplot_intf_feats 
from packages.classifiers.loocv_f1 import get_loocv_f1, get_loocv_pr_curve

def produce_cross_validated_scores(frame, target, feats, clf, ax, linestyle_iter, cs):
    print(f"Producing cross-validated scores for {feats}")
    get_loocv_f1(frame[feats], frame[target], LogisticRegression)
    precisions, recalls, thresholds, probs = get_loocv_pr_curve(frame[feats], frame[target], clf)
    auc_val = auc(recalls, precisions)

    label = ' + '.join(feats)
    label = f'{label} (AUC={auc_val:.2f})'

    ax.plot(recalls, precisions, label=label, linestyle=next(linestyle_iter), c=next(cs))
    return probs

def perform_regression(frame, feats, pairwise_vis_feats, dlda_fname, draw_pairwise_plots=True ):
    frame = frame.copy()
    for feat in feats:
        frame[[feat]] = StandardScaler().fit_transform(frame[[feat]])

    print(f"==============Regression with {feats}====================")
    transform = lambda x: 1 if x=='intf' else 0
    frame['type'] = frame['type'].apply(transform) 
    if draw_pairwise_plots:
        pairplot_intf_feats(frame, pairwise_vis_feats, f'controlled_intf_{dlda_fname}')

    
    print("Cross-validated scores; all")
    fig, ax = plt.subplots(1,1)
    fig.suptitle("Precision-recall curves for predicting emergence of intensifiers")

    linestyles = iter(["-", "--", "dashdot"]) 
    cs = iter(['firebrick', 'royalblue', 'darkgreen'])
    for feat in feats:
        frame[f'{feat}_prob'] = produce_cross_validated_scores(frame, 'type', [feat], LogisticRegression, ax, linestyles, cs)

    random_rs, random_prs = np.linspace(0, 1, 50), np.repeat(0.58, 50)
    auc_val = .48
    label='Random'

    label = f'{label} (AUC={auc_val:.2f})'

    ax.plot(random_rs, random_prs, label=label, linestyle=next(linestyles), c=next(cs))

    ax.set_xlabel("Recall", fontsize='large')
    ax.set_ylabel("Precision", fontsize='large')
    ax.legend(fontsize='large')
    ax.autoscale_view()
    plt.legend(loc='lower center')
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()
    store_pic_dynamic(plt, f'historical_pr', DYNAMIC_RESULTS_FOLDER)
    return frame

def get_ranking(frame, feat):
    ranking = frame[feat].rank(ascending=False)
    return ranking

def classify_half_thres(prob, typ):
    p = float(prob[1])
    if p > 0.5 and typ==1:
        return 'TP'
    elif p > 0.5 and typ==0:
        return 'FP'
    elif p <= 0.5 and typ==0:
        return 'TN'
    elif p <= 0.5 and typ==1:
        return 'FN'


def predict_intensifier_emergence(artifact_type):
    """Use logistic regression classifier to predict whether
    an adverb will acquire an intensifying sense. 
    """
    if artifact_type == USE_PROVIDED:
        artifacts_path = INTF_PRED_ARTIFACTS_PATH
    elif artifact_type == USE_GENERATED:
        artifacts_path = f'{DYNAMIC_ARTIFACTS_PATH}/{get_today_date_formatted()}'
    else:
        raise Exception("Provided unknown option to --produce_final_graph argument")

    centroid_frame, simadj_frame, = [load_pkl_from_path(f'{artifacts_path}/feat_{feat_name}_frame') for 
        feat_name in ['extremeness_single_centroid', 'simadjmod' ]]
    frames = [centroid_frame, simadj_frame]

    combined_frame = reduce(lambda f1, f2: f1.join(f2, rsuffix='_r'), frames)
    combined_frame_dropna = combined_frame.dropna()

    feats = ['prob_extreme', 'simadj']
    new_col_map = {"prob_extreme": "Extremeness",  "simadj": "SimAdjMod"}
    combined_frame_dropna = combined_frame_dropna.rename(columns=new_col_map)
    feats = list([new_col_map[feat] for feat in feats])
    combined_frame_dropna = perform_regression(combined_frame_dropna, feats, feats, '', draw_pairwise_plots=True)

    csv_frame = combined_frame_dropna.copy()

    for feat in feats:
        csv_frame[f'{feat}_clf'] = csv_frame[[f'{feat}_prob', 'type']].apply(lambda x: classify_half_thres(x[f'{feat}_prob'], x['type']), axis=1)

    feat_prob_columns = map_list(lambda x: f'{x}_prob', feats)
    feat_clf_columns = map_list(lambda x: f'{x}_clf', feats)
    csv_columns = feats +  feat_prob_columns + feat_clf_columns 
    csv_frame[feat_prob_columns] = csv_frame[feat_prob_columns].applymap(lambda x: f"{float(x[1]):.3f}")
    csv_frame[feats + feat_prob_columns] = csv_frame[feats + feat_prob_columns].applymap(lambda x: f"{float(x):.3f}")

    # csv_frame[['type', 'data_decs'] + csv_columns].to_csv(INTF_PRED_SPREADSHEET)

def main(args):
    if args.predict_intensifier_emergence:
        predict_intensifier_emergence(args.predict_intensifier_emergence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_intensifier_emergence', nargs='?', const=USE_PROVIDED, type=str)
    parser.add_argument('--inspect_csv', action='store_true')
    parser.add_argument('--get_num_intfs_added', action='store_true')
    parser.add_argument('--print_si_material', action='store_true')
    main(parser.parse_args())