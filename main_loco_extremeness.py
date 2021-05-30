from packages.pkl_operations.pkl_io import load_pkl_from_path, write_to_file, write_csv
from packages.utils.data_storage import get_save_path
from packages.classifiers.prototype_log_regression_classifier import PrototypeLogisticRegressionClassifier
from packages.utils.util_functions import map_list, list_diff
from packages.adverbs.visualizations import pr_curve, pr_curve_multiple 
from packages.adverbs.classify_ea_infra import build_pmi_frame
from packages.adverbs.constants import *
from packages.data_management.load_coca import compute_pmi_coca, compute_pmi_deps_coca
from packages.pkl_operations.pkl_io import store_results_dynamic, store_pic_dynamic
from packages.adverbs.visualizations import plot_loco_results

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, precision_recall_fscore_support, auc
from sklearn.preprocessing import normalize, StandardScaler

import argparse

np.random.seed(0)

def get_embed_frame(ws, w_to_embeds):
    """Returns a dataframe where the indices are the {ws}
    and the embeds are taken from the dictionary w_to_embeds.

    Args:
        ws ([type]): [description]
        w_to_embeds ([type]): [description]
    """
    embeds = []
    for w in ws:
        embeds.append(w_to_embeds[w])
    # TODO: normalize here.
    embeds = normalize(np.array(embeds))
    embed_frame = pd.DataFrame(data=embeds)
    embed_frame['word'] = ws
    embed_frame.set_index('word', inplace=True)
    return embed_frame


def gen_leave_one_out_results(model_cons, get_feat_frame, feats, method_name, pr_curve_fname, fig_title):
    """Evaluate a model using leave-one-cluster-out for the task of extremeness classification.

    Args:
        model_cons (() => Model): Constructor for a model. The object should have a fit, predict_proba, and predict method (like scikit-learn).
        get_feat_frame (([str]) => pd.DataFrame): Function that takes a list of adjectives and returns a DataFrame with features used to fit the model
        feats ([str]): Features to use to fit the model
        method_name (str): The name of the method used for evaluation (either DSM centroid or Distrib. test).
        pr_curve_fname (str): Name of the precision-recall curve (for saving).
        fig_title (str): Title of the precision-recall curve.
    """

    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET, index_col=1)
    cluster_frame = pd.read_csv(EXTR_ADJ_CLUSTER_SPREADSHEET, index_col=0)

    left_out_probs = []
    ys = []
    preds = []
    num_cluster_aucs = []
    for num_clusters in range(2, 15):
        for cluster_num in range(num_clusters):
            train_extr_cluster_frame = cluster_frame[cluster_frame[f'cluster_label_{num_clusters}'] != cluster_num]
            test_extr_cluster_frame = cluster_frame[cluster_frame[f'cluster_label_{num_clusters}'] == cluster_num]
            train_extr_adjs = train_extr_cluster_frame.index.values
            test_extr_adjs = test_extr_cluster_frame.index.values

            test_adj_frame = seed_frame.loc[test_extr_adjs]
            test_non_extr_adjs = test_adj_frame['Non-extreme adj.'].values
            test_embed_frame = get_feat_frame(list(test_extr_adjs) + list(test_non_extr_adjs))
            test_embed_frame['is_extr'] = map_list(lambda x: 1, test_extr_adjs) + map_list(lambda x: 0, 
                                                                                            test_non_extr_adjs)

            train_adj_frame = seed_frame.loc[train_extr_adjs]
            train_non_extr_adjs = train_adj_frame['Non-extreme adj.'].values
            train_embed_frame = get_feat_frame(list(train_extr_adjs) + list(train_non_extr_adjs))
            train_embed_frame['is_extr'] = map_list(lambda x: 1, train_extr_adjs) + map_list(lambda x: 0, 
                                                                                            train_non_extr_adjs)

            assert set(train_extr_adjs).intersection(set(test_extr_adjs)) == set([])
            model = model_cons().fit(train_embed_frame[feats], train_embed_frame['is_extr'])
            left_out_probs.extend(model.predict_proba(test_embed_frame[feats])[:,1])
            preds += list(model.predict(test_embed_frame[feats]))
            ys += list(test_embed_frame['is_extr'])
        precisions, recalls, thresholds = precision_recall_curve(ys, left_out_probs)
        auc_output = auc(recalls, precisions)
        num_cluster_aucs.append(auc_output)
        pr_curve(precisions, recalls, method_name, pr_curve_fname, fig_title) 
        precisions, recalls, f1s, _ = precision_recall_fscore_support(ys, preds, average='binary')
        print(f"AUC: {auc_output}")
        print(f"Precision for positive class at 0.5 threshold: {precisions}")
        print(f"Recall for positive class at 0.5 threshold: {recalls}")
        print(f"F1 for positive class at 0.5 threshold: {f1s}")
        print(f"Accuracy score: {accuracy_score(ys, preds)}")
    print(f"{num_cluster_aucs}")


def generate_clusters():
    """Generate clusters for the extreme adjectives using K-means clustering. 
    """
    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET)
    extreme_adjs = list(seed_frame['Extreme adj.'].values)

    w_to_embeds = load_pkl_from_path(f'{EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_WORD_TO_EMBEDS_FNAME}')
    extr_embed_frame = get_embed_frame(extreme_adjs, w_to_embeds)
    for num_clusters in range(2, 15):
        feats = np.arange(300)
        clusterer = KMeans(n_clusters=num_clusters)
        extr_embed_frame[f'cluster_label_{num_clusters}'] = clusterer.fit_predict(extr_embed_frame[feats])
    write_csv('extr_adj_clusters', extr_embed_frame[[f'cluster_label_{i}' for i in range(2,15)]])

def gen_leave_one_out_centroid():
    """Evaluate DSM centroid method using leave-one-cluster-out method
    """
    model_cons = PrototypeLogisticRegressionClassifier 
    w_to_embeds = load_pkl_from_path(f'{EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_WORD_TO_EMBEDS_FNAME}')
    feats = np.arange(300)

    def get_feat_frame(adjs):
        """Get the pandas DataFrame representing the features for each adjective

        Args:
            adjs ([str]): List of adjectives (extreme or non-extreme)
        """
        feat_frame = get_embed_frame(adjs, w_to_embeds)
        return feat_frame
    feats = np.arange(300)
    gen_leave_one_out_results(model_cons, get_feat_frame, feats, 'DSM Centroid', 'centroid_loco', 'Extremeness classification performance for DSM centroid')

def gen_leave_one_out_dist_test():
    """Evaluate Distrib. test method using leave-one-cluster-out method
    """
    model_cons = LogisticRegression 

    dms_2adjs_2counts = load_pkl_from_path(f"{APPENDIX_EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_NPMI_DM_ADJ_COLLOCATION_FREQ}")
    freq_dms_and_adjs = load_pkl_from_path(f"{APPENDIX_EXTR_CLF_ARTIFACTS_PATH}/{EXTR_CLF_NPMI_DM_ADJ_FREQ}")

    VOCAB_SIZE = 218851
    get_pmi_triples = lambda adjs: compute_pmi_coca(set(TOTALITY_MODS), 
                        adjs,
                        set(NON_EA_MODS),
                        dms_2adjs_2counts, 
                        freq_dms_and_adjs, 
                        VOCAB_SIZE
                    )

    def get_feat_frame(adjs):
        """Get the pandas DataFrame representing the features for each adjective

        Args:
            adjs ([str]): List of adjectives (extreme or non-extreme)
        """
        pmi_triples = get_pmi_triples(adjs)
        pmi_frame = build_pmi_frame(pmi_triples)
        return pmi_frame 
    feats = ['pmi_very', 'pmi_edm']
    gen_leave_one_out_results(model_cons, get_feat_frame, feats, 'Dist. test', 'dist_test_loco', 'Extremeness classification performance for Dist. test')


def compute_pmi_deps_coca_final():
    """Compute the required artifacts for calculating NPMI between
    extreme degree modifiers and adjectives.
    """
    seed_frame = pd.read_csv(EXTR_CLF_SPREADSHEET, index_col=1)
    extr_adjs = list(seed_frame['Extreme adj.'].values)
    nonextr_adjs = list(seed_frame['Non-extreme adj.'].values)
    compute_pmi_deps_coca(TOTALITY_MODS, set(extr_adjs).union(nonextr_adjs), set(NON_EA_MODS), 'query_ea')

def produce_supp_graph():
    """Produce main figure of Appendix A. Scores were generated from `gen_leave_one_out_centroid`
    and `gen_leave_one_out_dist_test`.
    """
    dist_test_aucs = [0.6787053784139941, 0.7006602025001032, 0.7196425705293841, 0.7264992786233386, 0.7279810688607387, 0.735211577157231, 0.7391620664191785, 0.7424551217358029, 0.744547118890123, 0.7462032178483676, 0.7470790984710051, 0.7477819649454344, 0.7485772939514082][:-2]
    centroid_aucs = [0.783994197668761, 0.8022824691267066, 0.8274006595739404, 0.8446694510975259, 0.8539680948101284, 0.8611348902666731, 0.8673677510331582, 0.8731568176307785, 0.8763207565027576, 0.8800770442321334, 0.8838125261131143, 0.8863090990318512, 0.8887080208666623][:-2]
    plot_loco_results(centroid_aucs, dist_test_aucs)


def main(args):
    if args.gen_leave_one_out_centroid:
        gen_leave_one_out_centroid()
    elif args.generate_clusters:
        generate_clusters()
    elif args.gen_leave_one_out_dist_test:
        gen_leave_one_out_dist_test()
    elif args.compute_pmi_deps_coca_final:
        compute_pmi_deps_coca_final()
    elif args.produce_supp_graph:
        produce_supp_graph()

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_leave_one_out_centroid', action='store_true')
    parser.add_argument('--gen_leave_one_out_dist_test', action='store_true')
    parser.add_argument('--eval_optimal_clusters_vbem', action='store_true')
    parser.add_argument('--generate_clusters', action='store_true')
    parser.add_argument('--compute_pmi_deps_coca_final', action='store_true')
    parser.add_argument('--produce_supp_graph', action='store_true')
    main(parser.parse_args())