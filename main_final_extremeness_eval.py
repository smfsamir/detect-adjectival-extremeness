"""This module contains the final evaluation results for the classification of 
extreme adjectives (Section 3 of the paper). 
"""
from packages.utils.data_storage import get_save_path
from packages.adverbs.constants import *
from packages.utils.util_functions import compute_npmi, get_artifacts_path, get_today_date_formatted
from packages.synchronic_embeddings.synchronic_w2v import get_embeddings
from packages.pkl_operations.pkl_io import load_pkl_from_path, store_results_dynamic, store_pic_dynamic
from packages.utils.calculations import cos
from packages.data_management.load_coca import compute_pmi_coca, compute_pmi_deps_coca
from packages.classifiers.loocv_f1 import get_loocv_pr_curve, get_loocv_f1
from packages.classifiers.prototype_log_regression_classifier import PrototypeLogisticRegressionClassifier
from packages.adverbs.classify_ea_infra import build_pmi_frame 
from packages.adverbs.visualizations import pr_curve, pr_curve_multiple 

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import auc
from collections import defaultdict
import numpy as np

import argparse
import sys
import seaborn as sns
import pandas as pd 
from functools import reduce, partial
import numpy as np
import matplotlib.pyplot as plt

def classification_collocation(frame, use_disassoc=True):
    """Performs binary classification of extreme adjectives using the NPMI (with extreme degree modifiers + non-extreme degree modifiers) 
    + logistic regression classifier.

    Args:
        frame (pd.DataFrame): |word|y|; adjectives (word) with their status (y; extreme or not)
        use_disassoc (bool, optional): Whether or not to use the (dis)association with non-extreme-degree modifiers as a feature for the logistic 
            regression classfier. Defaults to True.

    Returns:
        (np.array, np.array, np.array ): Precisions, recalls, and thresholds, as estimated by sklearn.metrics:precision_recall_curve.
    """
    frame['pmi_edm'] = StandardScaler().fit_transform(frame[['pmi_edm']])
    frame['pmi_very'] = StandardScaler().fit_transform(frame[['pmi_very']])

    if use_disassoc:
        prs, rs, ts, probs = get_loocv_pr_curve(frame[['pmi_edm', 'pmi_very']], frame['y'].values, LogisticRegression)
    else:
        prs, rs, ts, probs = get_loocv_pr_curve(frame[['pmi_edm']], frame['y'].values, LogisticRegression)
    return prs, rs, ts, probs

def classify_w_single_centroid(pos_ws, neg_ws, w_to_embeds):
    """Performs binary classification of extreme adjectives using a prototype + logistic regression classifier.

    Args:
        pos_ws ([str]): 
        neg_ws ([str]): [description]
        w_to_embeds ({str: np.array}): Mapping of adjectives to their word2vec embeddings.

    Returns:
        (np.array, np.array, np.array ): Precisions, recalls, and thresholds, as estimated by sklearn.metrics:precision_recall_curve.
    """
    ys = [1 for w in pos_ws] + [0 for w in neg_ws]
    embeds = []
    for w in pos_ws + neg_ws:
        embeds.append(w_to_embeds[w])
    embeds = normalize(np.array(embeds))
    embed_frame = pd.DataFrame(data=embeds)
    embed_frame['word'] = pos_ws + neg_ws
    embed_frame.set_index('word', inplace=True)
    embed_frame['y'] = np.array(ys)

    dims = np.arange(300)
    return get_loocv_pr_curve(embed_frame[dims], embed_frame['y'], PrototypeLogisticRegressionClassifier)

TOTALITY_MODS = ['totally', 'absolutely', 'simply', 'positively', 'downright', 'outright'] 

def evaluate_adjs_npmi():
    """Evaluates the NPMI method for classifying extreme adjectives. 
    The results are stored as the precision and recall elements from the 
    sklearn.metrics:precision_recall_curve method.

    The results are stored in f'{DYNAMIC_ARTIFACTS_PATH}/{date.today}/{EXTR_CLF_NPMI_AUC_ITEMS}'.
    """
    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET)
    extr_adjs = list(seed_frame['Extreme adj.'].values)
    nonextr_adjs = list(seed_frame['Non-extreme adj.'].values)
    dms_2adjs_2counts = load_pkl_from_path(f"{EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_NPMI_DM_ADJ_COLLOCATION_FREQ}")
    freq_dms_and_adjs = load_pkl_from_path(f"{EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_NPMI_DM_ADJ_FREQ}")

    pmi_triples = compute_pmi_coca(set(TOTALITY_MODS), 
                        reduce(lambda s1, s2: s1.union(s2), [set(extr_adjs), set(nonextr_adjs)]), 
                        set(NON_EA_MODS), 
                        dms_2adjs_2counts, 
                        defaultdict(int, freq_dms_and_adjs), 
                        COCA_VOCAB_SIZE
                    )

    pmi_frame = build_pmi_frame(pmi_triples)
    pmi_frame['y'] = pmi_frame.apply(lambda row: 1 if row.name in extr_adjs else 0, axis=1)

    prs, rs, ts, probs  = classification_collocation(pmi_frame)

    pr_and_r = [prs, rs]
    
    store_results_dynamic(pr_and_r, EXTR_CLF_NPMI_AUC_ITEMS, 'data/artifacts')

def produce_final_graph(artifact_type):
    """Produces the precision recall curves from Section 3 of the paper 
    (classification results from the DSM Centroid and Distrib. Test method).
    """
    if artifact_type == USE_PROVIDED:
        artifacts_path = EXTR_CLF_ARTIFACTS_PATH
    elif artifact_type == USE_GENERATED:
        artifacts_path = f'{DYNAMIC_ARTIFACTS_PATH}/{get_today_date_formatted()}'
    else:
        raise Exception("Provided unknown option to --produce_final_graph argument")

    npmi_disassoc_prs, npmi_disassoc_rs = load_pkl_from_path(f'{artifacts_path}/{EXTR_CLF_NPMI_AUC_ITEMS}')
    cent_prs, cent_rs = load_pkl_from_path(f'{artifacts_path}/{EXTR_CLF_CENTROID_AUC_ITEMS}')
    random_rs, random_prs = np.linspace(0, 1, len(npmi_disassoc_rs)), np.repeat(0.5, len(npmi_disassoc_prs))
    all_prs = [cent_prs, npmi_disassoc_prs, random_prs]
    all_recalls = [cent_rs, npmi_disassoc_rs,  random_rs]
    pr_curve_multiple(all_prs, all_recalls, ['DSM Centroid', 'Dist. test (E + O)',  'Random'], 'Extremeness classification precision-recall curves', EXTR_CLF_PRECISION_RECALL_FIG_FNAME)

def eval_single_centroid_model():
    """Evaluates the DSM Centroid method for classifying extreme adjectives. 
    The results are stored as the precision and recall elements from the 
    sklearn.metrics:precision_recall_curve method.

    The results are stored in f'{DYNAMIC_ARTIFACTS_PATH}/{date.today}/{EXTR_CLF_CENTROID_AUC_ITEMS}'
    """
    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET)
    extreme_adjs = list(seed_frame['Extreme adj.'].values)
    nonextr_adjs = list(seed_frame['Non-extreme adj.'].values)

    w_to_embeds = load_pkl_from_path(f'{EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_WORD_TO_EMBEDS_FNAME}')
    prs, rs, ts, probs = classify_w_single_centroid(extreme_adjs, nonextr_adjs, w_to_embeds)
    pr_and_r = [prs, rs]

    pr_curve_elems = [rs, prs, ts]
    store_results_dynamic(pr_and_r, EXTR_CLF_CENTROID_AUC_ITEMS, DYNAMIC_ARTIFACTS_PATH)

def compute_npmi_artifacts():
    """Compute the plug-in probability estimates, measured from the COCA corpus, required to calculate the NPMI
    between a degree modifier and an adjective. 
    """
    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET)
    extr_adjs = list(seed_frame['Extreme adj.'].values)
    nonextr_adjs = list(seed_frame['Non-extreme adj.'].values)
    compute_pmi_deps_coca(TOTALITY_MODS, set(extr_adjs).union(nonextr_adjs), set(NON_EA_MODS), 'query_ea')

def main(args):
    if args.evaluate_adjs_npmi:
        evaluate_adjs_npmi()
    elif args.compute_npmi_artifacts: 
        compute_npmi_artifacts()
    elif args.produce_final_graph: 
        produce_final_graph(args.produce_final_graph)
    elif args.eval_single_centroid_model:
        eval_single_centroid_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate_adjs_npmi', action='store_true')
    parser.add_argument('--eval_single_centroid_model', action='store_true')
    parser.add_argument('--produce_final_graph', nargs='?', const=USE_PROVIDED, type=str)
    parser.add_argument('--compute_npmi_artifacts', action='store_true')
    main(parser.parse_args())