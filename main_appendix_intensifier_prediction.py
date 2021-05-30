from packages.pkl_operations.pkl_io import load_pkl_from_path, store_results_dynamic, store_pic_dynamic
from packages.adverbs.compute_sim_adj_nc import compute_simadj_per_dec
from packages.adverbs.get_source_adj import get_closest_adjs
from packages.adverbs.util_functions import load_full_intensifiers, load_control_advs
from packages.pkl_operations.pkl_io import store_results_dynamic
from packages.adverbs.diachronic_exemplar import single_centroid_prob_extreme
from packages.classifiers.loocv_f1 import get_loocv_f1, get_loocv_pr_curve
from packages.adverbs.constants import *

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc


def compute_simadjmod_avgd():
    """Compute SimAdjMod for every control adverb and intensifier for every decade from
       [1850,..., 1990]. Average the values over every decade for the adverbs.
    """
    adv_2dec_2adjs_modded = load_pkl_from_path(ADV_TO_DEC_TO_ADJS_MODDED)
    adj_2dec_2freq_modded = load_pkl_from_path(ADJ_TO_DEC_TO_FREQ_MODDED)
    adj_2_freq = load_pkl_from_path(ADJ_TO_FREQ)

    time_range = range(1850, 2000, 10)
    INTENSIFIERS = load_full_intensifiers()
    CONTROL_WORDS = load_control_advs()
    frame = pd.DataFrame({
        'type': ['intf' for intf in INTENSIFIERS] + ['ctrl' for ctrl in CONTROL_WORDS],
        'adv': INTENSIFIERS + CONTROL_WORDS
    })
    frame.set_index('adv', inplace=True)
    for dec in time_range: 
        sim_adjs_modded = compute_simadj_per_dec(INTENSIFIERS + CONTROL_WORDS, 
                                                    adv_2dec_2adjs_modded, 
                                                    adj_2_freq, 
                                                    adj_2dec_2freq_modded, 
                                                    dec, True)
        frame[f'sim_adj_{dec}'] = sim_adjs_modded
    store_results_dynamic(frame, 'simadjmod_avgd_results_frame', DYNAMIC_ARTIFACTS_PATH)

def calculate_extremeness_per_dec():
    """Compute Extremeness (proximity to extreme centroid) for every control adverb and intensifier for every decade from
       [1850,..., 1990]. Average the values over every decade for the adverbs.
    """
    unwrap = lambda opt_words: list(map(lambda opt_word: opt_word.val, opt_words))
    seed_frame = pd.read_csv(f'{EXTR_CLF_SPREADSHEET}')
    extreme_adjs = list(seed_frame['Extreme adj.'].values)

    INTENSIFIERS = load_full_intensifiers()
    CONTROL_WORDS = load_control_advs()

    opt_intf_adj_bases = get_closest_adjs(INTENSIFIERS, 1, lambda x: print(f"{x} has no corresponding pertainym"))
    opt_ctrl_adj_bases = get_closest_adjs(CONTROL_WORDS, 1, lambda x: print(f"{x} has no corresponding pertainym"))

    intf_adj_bases = unwrap(opt_intf_adj_bases)
    ctrl_adj_bases = unwrap(opt_ctrl_adj_bases)
    all_bases = intf_adj_bases + ctrl_adj_bases

    time_range = range(1850, 2000, 10) 
    word_frame = pd.DataFrame({
        'type': ['intf' for intf in intf_adj_bases] + ['ctrl' for ctrl in ctrl_adj_bases], 
        'adv': INTENSIFIERS + CONTROL_WORDS,
        'adjs': all_bases 
        }
    )
    for dec in time_range:
        print(f"Processing extremeness values for {dec}s")
        dec_word_frame = word_frame.copy()
        dec_word_frame[f'{dec}_discrim_pos'] = single_centroid_prob_extreme(dec_word_frame, extreme_adjs, dec)
        word_frame = word_frame.merge(dec_word_frame[[f'{dec}_discrim_pos', 'adjs']], left_on='adjs', right_on='adjs')
    word_frame['mean_discrim_pos'] = word_frame[[f'{dec}_discrim_pos' for dec in time_range]].mean(axis=1, skipna=True)
    store_results_dynamic(word_frame, f'extremeness_avgd_frame_single_centroid' , DYNAMIC_ARTIFACTS_PATH)

# NOTE: mean_discrim_pos refers to mean_extremeness
def classify_intensifiers():
    extr_avgd_frame = load_pkl_from_path(EXTREMENESS_AVGD_FRAME)
    sam_frame = load_pkl_from_path(SIMADJMOD_AVGD_FRAME)
    sam_frame_dropna = sam_frame.copy()

    time_range = range(1850, 2000, 10)
    sam_frame = sam_frame.replace([np.inf, -np.inf], np.nan) 
    sam_frame['mean_simadj'] = sam_frame[[f'sim_adj_{dec}' for dec in time_range]].mean(axis=1, skipna=True)

    sam_frame_dropna = sam_frame_dropna.dropna()
    sam_frame_dropna['mean_simadj_dropna_rows'] = sam_frame_dropna[[f'sim_adj_{dec}' for dec in time_range]].mean(axis=1)

    sam_frame = sam_frame.loc[sam_frame['mean_simadj'].dropna().index]
    extr_avgd_frame = extr_avgd_frame.set_index('adv')
    sam_extr_frame = sam_frame.join(extr_avgd_frame, rsuffix='_r')

    sam_extr_frame = sam_extr_frame.loc[sam_extr_frame[['mean_simadj', 'mean_discrim_pos']].dropna().index]
    sam_extr_frame.y = sam_extr_frame['type'].apply(lambda t: 1 if t =='intf' else 0 )
    sam_extr_frame[['mean_discrim_pos']] = StandardScaler().fit_transform(sam_extr_frame[['mean_discrim_pos']])
    sam_extr_frame[['mean_simadj']] = StandardScaler().fit_transform(sam_extr_frame[['mean_simadj']])

    clf = lambda : LogisticRegression(solver='lbfgs')

    linestyles = iter(["-", "--", "dashdot"]) 
    cs = iter([ 'firebrick', 'royalblue', 'darkgreen'])

    fig, ax = plt.subplots(1,1)
    pred_discrim_pos = ['mean_discrim_pos']
    get_loocv_f1(sam_extr_frame[pred_discrim_pos], sam_extr_frame.y, clf)
    precisions, recalls, thresholds, prediction_probs = get_loocv_pr_curve(sam_extr_frame[pred_discrim_pos], sam_extr_frame.y, clf)
    auc_val = auc(recalls,precisions)
    plt.plot(recalls, precisions, label=f'Extremeness (AUC={auc_val:.2f})', linestyle=next(linestyles), c=next(cs))

    ###################### simadjmod 
    print("Cross-validated scores; simadj")
    pred_simadj = ['mean_simadj']
    get_loocv_f1(sam_extr_frame[pred_simadj], sam_extr_frame.y, clf)
    precisions, recalls, thresholds, prediction_probs = get_loocv_pr_curve(sam_extr_frame[pred_simadj], sam_extr_frame.y, clf)
    auc_val = auc(recalls,precisions)
    plt.plot(recalls, precisions, label=f'SimAdj (AUC={auc_val:.2f})', linestyle=next(linestyles), c=next(cs))
    plt.title('Precision-recall curves for classifying intensifiers')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    random_rs, random_prs = np.linspace(0, 1, 50), np.repeat(0.55, 50)
    auc_val = .55
    label='Random'

    label = f'{label} (AUC={auc_val:.2f})'

    plt.plot(random_rs, random_prs, label=label, linestyle=next(linestyles), c=next(cs))

    ax.set_xlabel("Recall", fontsize='large')
    ax.set_ylabel("Precision", fontsize='large')
    ax.legend(fontsize='large')
    ax.autoscale_view()
    plt.legend()
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()

    store_pic_dynamic(plt, f'pr_avgd', 'data/artifacts', True)


def main(args):
    if args.compute_simadjmod_avgd:
        compute_simadjmod_avgd()
    elif args.calculate_extremeness_per_dec:
        calculate_extremeness_per_dec()
    elif args.classify_intensifiers:
        classify_intensifiers()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_simadjmod_avgd', action='store_true')
    parser.add_argument('--calculate_extremeness_per_dec', action='store_true')
    parser.add_argument('--classify_intensifiers', action='store_true')
    main(parser.parse_args())