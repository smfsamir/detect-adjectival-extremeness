"""This module corresponds to Section 4.1 and 4.2 of the paper. It serves to set up
the analysis reported in Section 4.3 of the paper. 
"""

import argparse

from packages.pkl_operations.pkl_io import *
from packages.ngrams.load_token_dec_counts import save_counts_per_dec
from packages.adverbs.constants import *
from packages.adverbs.util_functions import get_dec_i_range, load_control_advs
from packages.utils.util_functions import conv_table_to_latex_string, roundup, round_down, map_list, map_unpack_opt
from packages.adverbs.get_source_adj import get_closest_adjs
from packages.adverbs.compute_sim_adj import compute_sim_adj_control
from packages.adverbs.diachronic_exemplar import single_centroid_prob_extreme, centroid_prob_extreme_controlled

import pandas as pd
import numpy as np
from itertools import chain



NUM_DECS_TU = 3
CONTROL_WORDS = load_control_advs()

def get_data_decs(intfs_to_attd, include_attested_year=True):
    """Returns {intf1: [1800], intf2: [1800,1810]}.
    That is, intensifiers mapped to {min(0, attd_date - NUM_DECS_TU): attd_date}

    Args:
        intfs_to_attd ([type]): [description]
    """
    intf_to_dd = {}
    for intf, attd in intfs_to_attd.items():
        attd_dec = roundup(attd) if include_attested_year else roundup(attd - 10)
        start_range = max(1800, attd_dec - (NUM_DECS_TU * 10)) 
        end_range = attd_dec if include_attested_year else round_down(attd)
        up_to_attd = range(start_range, end_range, 10)
        intf_to_dd[intf] = list(up_to_attd)
    return intf_to_dd

def match_to_control(intfs_to_attd, strs_to_freq, unique_only=False, match_to_adj=False, include_attested_year=True):
    """Returns a frame with intensifiers matched to control adverbs

    Args:
        intfs_to_attd ({str: int}): Date that the intensifier is attested.
        strs_to_freq ({str: [int]}): Strings wto their frequency {INTENSIFIERS}\cup{CTRL} over 20 decades; for both adverbial and adjectival forms
        unique_only (bool, optional): Each intensifier mapping is to a unique control adjective.
        match_to_adj (bool, optional): Perform the frequency match with the adjectival bases rather than the adverbial forms

    Returns:
        pd.DataFrame: Frame with intensifiers mapped to control adverbs as well as data timeframe. See
            *Section 4.1: Prediction  timeframe* for formal definition of this time frame.
    """
    intf_to_data_decs = get_data_decs(intfs_to_attd, include_attested_year)
    intfs = []
    ctrls = []
    data_decs_arr = []
    matched = set([])
    if match_to_adj:
        intf_adjs = get_closest_adjs(intfs_to_attd.keys())
        intf_strs = list(map_unpack_opt(intf_adjs))
        ctrl_strs = list(map_unpack_opt(get_closest_adjs(CONTROL_WORDS)))
        adj_to_adv = {adj: adv for (adj, adv) in zip(chain(intf_strs, ctrl_strs), chain(intfs_to_attd.keys(), CONTROL_WORDS))}
    else:
        intf_strs = intfs_to_attd.keys()
        ctrl_strs = CONTROL_WORDS
        
    for intf in intf_strs:
        if match_to_adj:
            data_decs = intf_to_data_decs[adj_to_adv[intf]]
        else:
            data_decs = intf_to_data_decs[intf]
        lower_range_i, upper_range_i = get_dec_i_range(data_decs)

        intf_freq = sum(strs_to_freq[intf][lower_range_i: upper_range_i])

        ctrl_freqs = []
        match_cands = set(ctrl_strs).difference(matched) if unique_only else ctrl_strs 
        match_cands = list(match_cands)

        for ctrl in match_cands: 
            ctrl_freq = sum(strs_to_freq[ctrl][lower_range_i: upper_range_i])
            ctrl_freqs.append(ctrl_freq)
        ctrl_freqs = np.array(ctrl_freqs) 
        diff_freqs = [abs(ctrl_freq - intf_freq) for ctrl_freq in ctrl_freqs]
        ctrl_freqs_arg_is = np.argsort(diff_freqs)

        if match_to_adj:
            intfs.append(adj_to_adv[intf])
            ctrl_match = match_cands[ctrl_freqs_arg_is[0]]
            ctrls.append(adj_to_adv[ctrl_match])
        else:
            intfs.append(intf)
            ctrl_match = match_cands[ctrl_freqs_arg_is[0]]
            ctrls.append(ctrl_match)
        matched.add(ctrl_match)
        data_decs_arr.append(data_decs)

    frame = pd.DataFrame({"ctrls": ctrls, "data_decs": data_decs_arr}, index=pd.Index(intfs, name='intensifier'))
    return frame


def conv_match_to_long_form(frame):
    """

    Args:
        frame (pd.DataFrame): Dataframe with intensifiers as indices that are mapped to 
            frequency-matched control adverbs. 

    Returns:
        pd.DataFrame: Return a DataFrame that is twice as long as the input frame. Each 
            control adverb and intensifier gets its own row. In addition, each row
            also contains the adjectival base of the intensifier. This long-form DataFrame
            facilitates classification (since we do classification on each row).
    """
    intfs = frame.index.values
    ctrls = frame.ctrls.values
    data_decs = frame.data_decs.values
    data_dec_exp = list(data_decs) * 2 
    types = ['intf'] * len(intfs) + ['ctrl'] * len(ctrls)
    concat_intf_ctrls = np.concatenate((intfs, ctrls))
    adjs = [opt.val for opt in get_closest_adjs(concat_intf_ctrls)]
    
    ctrls_annot = [f'{ctrl}_{intfs[i]}' for i, ctrl in enumerate(ctrls)]
    long_frame = pd.DataFrame({ 'type': types, 'data_decs': data_dec_exp, 'adjs': adjs}, index=pd.Index(np.concatenate((intfs, ctrls_annot)), name='adv'))
    return long_frame
    
def save_ngram_freq_advs_adjs():
    """Stores frequencies of control adverbs + intensifiers for every decade from 1800s to the 2000s.
    """
    frame_intensifier_dates = pd.read_csv("data/acl_artifacts/intensifier_dates.csv", dtype={"attested_year": int}, index_col="intensifier")
    intensifiers = list(frame_intensifier_dates.index.values)

    ctrl_advs = load_control_advs()
    intf_adjs = list(map(lambda x: x.val, get_closest_adjs(intensifiers)))
    ctrl_adjs = list(map(lambda x: x.val, get_closest_adjs(ctrl_advs)))
    save_counts_per_dec(intensifiers + ctrl_advs + intf_adjs + ctrl_adjs, ADV_FREQUENCY_PER_DEC)

def match_to_control():
    """Pair intensifiers with control adverbs based on frequency and the 
    attestation date of the intensifying sense of the intensifier. 
    """
    frame_intensifier_dates = pd.read_csv("{ACL_SPREADSHEETS_PATH}/{INTF_TO_ATTD_DATE_CSV}", dtype={"attested_year": int}, index_col="intensifier")
    intensifier_to_attd = frame_intensifier_dates.to_dict()['attested_year']
    exp_items_to_freq = load_pkl_from_path(f'{INTF_PRED_ARTIFACTS_PATH}/{ADV_FREQUENCY_PER_DEC}')

    years = sorted(list(intensifier_to_attd.values()))
    match_frame = match_to_control(intensifier_to_attd, exp_items_to_freq, False, False, False)
    store_results_dynamic(match_frame, INTENSIFIER_TO_CONTROL_MATCH_FRAME, DYNAMIC_ARTIFACTS_PATH)

def compute_extremeness():
    """Compute proximity to extreme centroid for every intensifier and matched control adverb.
    """
    match_frame = load_pkl_from_path(f'{INTF_PRED_ARTIFACTS_PATH}/{INTENSIFIER_TO_CONTROL_MATCH_FRAME}')
    seed_frame = pd.read_csv(f'{EXTR_CLF_SPREADSHEET}')
    extreme_adjs = list(seed_frame['Extreme adj.'].values)

    long_frame  = conv_match_to_long_form(match_frame)
    usable_pos_seeds = set(extreme_adjs).difference(set(long_frame.adjs.values))
    extr_frame = centroid_prob_extreme_controlled(long_frame, usable_pos_seeds)

    store_results_dynamic(extr_frame, INTF_EXTREMENESS_FEATURE_FRAME, DYNAMIC_ARTIFACTS_PATH)

def compute_simadj():
    """Compute SimAdjMod (Luo et al., 2019) for every intensifier and matched control adverb.
    """
    adv_2dec_2adjs_modded = load_pkl_from_path(ADV_TO_DEC_TO_ADJS_MODDED)
    adj_2dec_2freq_modded = load_pkl_from_path(ADJ_TO_DEC_TO_FREQ_MODDED)
    adj_2_freq = load_pkl_from_path(ADJ_TO_FREQ)

    match_frame = load_pkl_from_path(f'{INTF_PRED_ARTIFACTS_PATH}/{INTENSIFIER_TO_CONTROL_MATCH_FRAME}')
    long_frame = conv_match_to_long_form(match_frame)
    sam_frame = compute_sim_adj_control(long_frame, adv_2dec_2adjs_modded, adj_2dec_2freq_modded, adj_2_freq)

    store_results_dynamic(sam_frame, INTF_SIMADJMOD_FEATURE_FRAME, DYNAMIC_ARTIFACTS_PATH)

def main(args):
    if args.match_to_control: 
        match_to_control()
    elif args.save_ngram_freq_advs_adjs: 
        save_ngram_freq_advs_adjs()
    elif args.compute_extremeness: 
        compute_extremeness()
    elif args.compute_simadj: 
        compute_simadj()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_to_control', action='store_true')
    parser.add_argument('--compute_extremeness', action='store_true')
    parser.add_argument('--compute_simadj', action='store_true')
    parser.add_argument('--save_ngram_freq_advs_adjs', action='store_true')
    main(parser.parse_args())