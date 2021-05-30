import sys
import numpy as np
from functools import partial
import pandas as pd
import argparse

from packages.classifiers.prototype_log_regression_classifier import PrototypeLogisticRegressionClassifier
from packages.adverbs.visualizations import pr_curve_multiple
from packages.adverbs.classify_ea_infra import build_pmi_frame
from packages.utils.calculations import cos
from packages.adverbs.constants import TOTALITY_MODS, NON_EA_MODS
from packages.data_management.coca.load_freq import load_lemma_freqs_by_word_tag
from packages.data_management.load_coca import compute_pmi_coca, compute_pmi_deps_coca
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import label_binarize
from packages.data_management.load_adj_scales import load_scales
from packages.utils.util_functions import map_list
from packages.synchronic_embeddings.synchronic_w2v import get_embed_frame, get_embeddings
from packages.pkl_operations.pkl_io import write_csv, store_results_dynamic, store_pic_dynamic, load_pkl_from_path
from packages.adverbs.visualizations import pr_curve 

def get_non_extreme_adjs(scales_frame, extreme_adj_df):
    scale_frame = scales_frame.loc[(scales_frame['scale'])==(extreme_adj_df['scale'])]
    non_extreme_adj_frame = scale_frame[scale_frame['ranks'] > extreme_adj_df['ranks']]
    non_extreme_adj_frame = non_extreme_adj_frame.rename(columns={'adj': 'non_extreme_adj', 'ranks': 'non_extreme_rank'})
    return non_extreme_adj_frame[['non_extreme_adj', 'non_extreme_rank']]

def get_neg_seeds_adj_scales():
    extreme_adjs = set(pd.read_csv('acl_supp_extremeness_predictions.csv')['Extreme adj.'].values)
    scales_frame = load_scales()
    extreme_scales_frame = scales_frame.loc[scales_frame['adj'].isin(extreme_adjs)]
    non_extreme_scales_frame = scales_frame.loc[~scales_frame['adj'].isin(extreme_adjs)]
    non_extreme_scales_frame.rename(columns={'adj': 'cand_non_extreme_adj', 'ranks': 'cand_rank'}, inplace=True)
    extreme_non_extreme_cand_frame = extreme_scales_frame.merge(non_extreme_scales_frame, on='scale')
    print(extreme_non_extreme_cand_frame)
    extreme_non_extreme_cand_frame['select'] = extreme_non_extreme_cand_frame['ranks'] < extreme_non_extreme_cand_frame['cand_rank']
    extreme_non_extreme_cand_frame = extreme_non_extreme_cand_frame[extreme_non_extreme_cand_frame['select']]
    extreme_non_extreme_cand_frame.rename(columns={'adj': 'extreme_adj', 'ranks': 'extreme_adj_rank'}, inplace=True)
    write_csv('non_extreme_paired_adjs', extreme_non_extreme_cand_frame[['extreme_adj', 'scale', 'extreme_adj_rank', 'cand_non_extreme_adj', 'cand_rank']])

def _filter_superlatives_comparatives_noun_dom(adj_frame):
    """Filter NON-EXTREME adjectives that meet any of the following conditions:
        (1) Is noun-dominant
        (2) is a superlative
        (3) is a comparative

    Args:
        adj_frame (pd.DataFrame): |adjective(index)|scale|status
    """
    non_extreme_adj_frame = adj_frame.loc[adj_frame['status'] == 'cand_non_extreme_adj']
    freq_frame = load_lemma_freqs_by_word_tag(list(non_extreme_adj_frame.index.values))
    def _filter_noun_dominant(freq_frame):
        lemmas_with_j_pos_frame = freq_frame.loc[freq_frame['PoS']=='j']
        lemmas_with_j = lemmas_with_j_pos_frame.lemma.values
        lemmas_non_j_pos_frame = freq_frame.loc[(freq_frame['PoS']!= 'j') & (freq_frame['lemma'].isin(set(lemmas_with_j)))]
        lemmas_non_j = set(lemmas_non_j_pos_frame.lemma.values)
        lemmas_to_filter = []
        for i in range(len(lemmas_with_j)):
            curr_lemma = lemmas_with_j[i]
            if lemmas_with_j[i] in lemmas_non_j: 
                lemma_j_freq = lemmas_with_j_pos_frame.loc[lemmas_with_j_pos_frame['lemma']==curr_lemma]['freq'].values[0]
                lemma_non_j_freq = lemmas_non_j_pos_frame.loc[lemmas_non_j_pos_frame['lemma']==curr_lemma]['freq'].values
                if any(lemma_non_j_freq > lemma_j_freq):
                    lemmas_to_filter.append(curr_lemma)
                # TODO: fill in this line.
        return adj_frame[~adj_frame.index.isin(lemmas_to_filter)]
    
    def _filter_comparative(adj_frame):
        non_extreme_adj_frame = adj_frame.loc[adj_frame['status'] == 'cand_non_extreme_adj']
        non_extreme_lemmas = non_extreme_adj_frame.index.values
        comparatives = [lemma for lemma in non_extreme_lemmas if lemma[-2:] == 'er']
        comparatives.remove('clever')
        return adj_frame[~adj_frame.index.isin(comparatives)]

    def _filter_superlative(adj_frame):
        non_extreme_adj_frame = adj_frame.loc[adj_frame['status'] == 'cand_non_extreme_adj']
        non_extreme_lemmas = non_extreme_adj_frame.index.values
        superlatives = [lemma for lemma in non_extreme_lemmas if lemma[-3:] == 'est']
        return adj_frame[~adj_frame.index.isin(superlatives)]

    adj_frame = _filter_noun_dominant(freq_frame)
    adj_frame = _filter_comparative(adj_frame)
    adj_frame = _filter_superlative(adj_frame)
    return adj_frame

def prep_classification_spreadsheet():
    paired_adj_frame = pd.read_csv('results/spreadsheets/2021-03-22/non_extreme_paired_adjs.csv')
    melted_frame = pd.melt(paired_adj_frame, id_vars=['scale'], value_vars=['extreme_adj', 'cand_non_extreme_adj'],
                            var_name='status', value_name='adjective')
    melted_frame = melted_frame.drop_duplicates(subset=['adjective', 'scale'])
    melted_frame = melted_frame.set_index('adjective')
    melted_frame = _filter_superlatives_comparatives_noun_dom(melted_frame)
    print(melted_frame)

    adjs = list(set(melted_frame.index.values))
    adj_to_embeds = load_pkl_from_path('data/artifacts/2021-04-21/adj_scale_embeds')
    # adj_to_embeds = get_embeddings(adjs)
    store_results_dynamic(adj_to_embeds, 'adj_scale_embeds')
    embed_frame = get_embed_frame(adjs, adj_to_embeds)
    print(embed_frame)
    melted_frame = melted_frame.join(embed_frame)
    # this should be the final spreadsheet
    print(melted_frame)
    write_csv('adj_scales_classification_spreadsheet', melted_frame, 'data/spreadsheets')

def perform_leave_one_scale_out_classification_centroid():
    model_cons = PrototypeLogisticRegressionClassifier 
    adj_scales_frame = pd.read_csv('data/spreadsheets/2021-05-05/adj_scales_classification_spreadsheet.csv', index_col=0)
    feats = map_list(str, np.arange(300))

    def get_feat_frame(adjs):
        """Get the pandas DataFrame representing the features for each adjective

        Args:
            adjs ([str]): List of adjectives (extreme or non-extreme)
        """
        frame = adj_scales_frame.loc[adjs]
        return frame

    prs, rs, ts, probs = _leave_one_scale_out_loop(adj_scales_frame, model_cons, get_feat_frame, feats, 'DSM Centroid', 'loso_extr_centroid_pr_curve', 'Leave-one-scale-out classification performance')
    pr_and_r = [prs, rs]
    store_results_dynamic(pr_and_r, 'adj_scale_centroid_pr_and_r')
    
def _leave_one_scale_out_loop(adj_scales_frame, model_cons, get_feat_frame, feats,
                               feat_name, pr_curve_fname, fig_title):
    w2v_dims = map_list(str, np.arange(300))
    scales = set(adj_scales_frame.scale.values)
    prob_predictions = []
    label_predictions = []
    adj_scales_frame

    classes, bin_labels = np.unique(adj_scales_frame['status'], return_inverse=True)
    pos_index = list(classes).index('extreme_adj')
    adj_scales_frame['status_binary'] = bin_labels
    fps = []
    fns = []
    actual_labels = []

    for scale in scales: # basically like a leave-one-out loop
        train_frame = adj_scales_frame.loc[adj_scales_frame['scale'] != scale]
        test_frame = adj_scales_frame.loc[adj_scales_frame['scale'] == scale]
        train_frame = train_frame[~train_frame.index.isin(test_frame.index)]

        # TODO: drop duplicate indices as well.
        train_frame = train_frame[~train_frame.index.duplicated(keep='first')]
        test_frame = test_frame[~test_frame.index.duplicated(keep='first')]

        train_frame = get_feat_frame(train_frame.index.values)
        test_frame = get_feat_frame(test_frame.index.values)

        model = model_cons()
        model.fit(train_frame[feats], train_frame['status_binary'])
        iter_test_label_predictions = model.predict(test_frame[feats])
        iter_test_prob_predictions = model.predict_proba(test_frame[feats])[:, pos_index] 
        # NOTE: the predictions may not be arrays, possibly lists? 
        prob_predictions.extend(iter_test_prob_predictions)
        label_predictions.extend(iter_test_label_predictions)
        actual_labels.extend(list(test_frame['status_binary'].values))

        for i in range(len(iter_test_label_predictions)):
            test_item = test_frame.iloc[i]
            test_label_prediction = iter_test_label_predictions[i]
            if  test_item.status_binary != test_label_prediction:
                if test_item.status_binary == pos_index:
                    fns.append(test_item.name)
                else:
                    fps.append(test_item.name)

    precisions, recalls, f1s, _ = precision_recall_fscore_support(actual_labels, label_predictions, average='binary')
    print(f"Precision for positive class at 0.5 threshold: {precisions}")
    print(f"Recall for positive class at 0.5 threshold: {recalls}")
    print(f"F1 for positive class at 0.5 threshold: {f1s}")

    precisions, recalls, thresholds = precision_recall_curve(actual_labels, np.array(prob_predictions))
    pr_curve(precisions, recalls, feat_name, pr_curve_fname, fig_title)

    return (precisions, recalls, thresholds, prob_predictions)

def generate_adj_scale_dist_dependencies():
    adj_scales_frame = pd.read_csv('data/spreadsheets/2021-05-05/adj_scales_classification_spreadsheet.csv', index_col=0)
    adjs = (adj_scales_frame.index.values)
    compute_pmi_deps_coca(TOTALITY_MODS, set(adjs), set(['very', 'slightly', 'almost']), 'query_ea_adj_scales')

def perform_leave_one_scale_out_classification_dist_test():
    adj_scales_frame = pd.read_csv('data/spreadsheets/2021-05-05/adj_scales_classification_spreadsheet.csv', index_col=0)
    dms_2adjs_2counts = load_pkl_from_path("data/artifacts/2021-05-11/query_ea_adj_scales_coca_dms_2adjs_2counts")
    freq_dms_and_adjs = load_pkl_from_path("data/artifacts/2021-05-11/query_ea_adj_scales_coca_freq_dms_and_adjs")

    VOCAB_SIZE = 218851
    get_pmi_triples = lambda adjs: compute_pmi_coca(set(TOTALITY_MODS), 
                        adjs,
                        set(NON_EA_MODS),
                        dms_2adjs_2counts, 
                        freq_dms_and_adjs, 
                        VOCAB_SIZE
                    )

    model_cons = LogisticRegression 
    feats = ['pmi_very', 'pmi_edm']

    def get_feat_frame(adjs):
        """Get the pandas DataFrame representing the features for each adjective

        Args:
            adjs ([str]): List of adjectives (extreme or non-extreme)
        """
        subset_scales_frame = adj_scales_frame.loc[adjs]
        subset_scales_frame = subset_scales_frame[~subset_scales_frame.index.duplicated(keep='first')]
        pmi_triples = get_pmi_triples(adjs)
        pmi_frame = build_pmi_frame(pmi_triples)
        return subset_scales_frame.join(pmi_frame)

    prs, rs, _, _ = _leave_one_scale_out_loop(adj_scales_frame, model_cons, get_feat_frame, feats, 'Dist. test', 'loso_dist_test_pr_curve', 'Leave-one-scale-out classification performance')
    pr_and_r = [prs, rs]
    store_results_dynamic(pr_and_r, 'adj_scale_dist_test_pr_and_r')

def generate_supp_graph():
    cent_prs, cent_rs = load_pkl_from_path('results/2021-05-11/adj_scale_centroid_pr_and_r')
    npmi_disassoc_prs, npmi_disassoc_rs = load_pkl_from_path('results/2021-05-11/adj_scale_dist_test_pr_and_r')
    all_prs = [cent_prs, npmi_disassoc_prs]
    all_recalls = [cent_rs, npmi_disassoc_rs]
    pr_curve_multiple(all_prs, all_recalls, ['DSM Centroid', 'Dist. test (E + O)'], 'Leave-one-scale-out extremeness classification precision-recall curves', 'leave_one_scale_out_pr_curves')

def main(args):
    if args.get_neg_seeds_adj_scales:
        (get_neg_seeds_adj_scales())
    elif args.prep_classification_spreadsheet:
        prep_classification_spreadsheet()
    elif args.perform_leave_one_scale_out_classification_centroid:
        perform_leave_one_scale_out_classification_centroid()
    elif args.perform_leave_one_scale_out_classification_dist_test:
        perform_leave_one_scale_out_classification_dist_test()
    elif args.generate_adj_scale_dist_dependencies:
        generate_adj_scale_dist_dependencies()
    elif args.generate_supp_graph:
        generate_supp_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_neg_seeds_adj_scales', action='store_true')
    parser.add_argument('--generate_supp_graph', action='store_true')
    parser.add_argument('--prep_classification_spreadsheet', action='store_true')
    parser.add_argument('--generate_adj_scale_dist_dependencies', action='store_true')
    parser.add_argument('--perform_leave_one_scale_out_classification_centroid', action='store_true')
    parser.add_argument('--perform_leave_one_scale_out_classification_dist_test', action='store_true')
    main(parser.parse_args())
