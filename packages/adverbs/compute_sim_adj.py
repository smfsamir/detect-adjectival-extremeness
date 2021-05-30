import numpy as np
from functools import reduce, partial
from collections import defaultdict
from ..utils.util_functions import round_down, roundup
from ..data_management.load_histwords_embeddings import get_avg_hw_embeddings, load_avg_hw_embeddings_map
from .get_source_adj import get_closest_adjs
from ..utils.calculations import cos
from .constants import Vec
from ..adverbs.util_classes import Optional

def compute_freq_modded(adjs, adj_2dec_2freq_modded, decs):
    """Return a map of {adj: freq_modified} where freq_modified
    denotes the total frequency with which the adjective was modified by an adverb
    across all decades in {decs}


    Arguments:
        adjs {[type]} -- [description]
        adj_2dec_2freq_modded {[type]} -- [description]
        decs {[type]} -- [description]
    """
    count_modded = defaultdict(int)
    for dec in decs:
        for adj in adjs:
            if dec in adj_2dec_2freq_modded[adj]:
                count_modded[adj] += adj_2dec_2freq_modded[adj][dec]
    return count_modded

def compute_freq_total(adjs, adj_2_total_freq, decs):
    count_total = defaultdict(int)
    for dec in decs:
        for adj in adjs:
            count_total[adj] += adj_2_total_freq[adj][(dec - 1800) // 10]
    return count_total 

def compute_odds_modded(adj_2freq_modded, adj_2_total_freq):
    """Compute the odds that the adjective is modified by 
    an adverb. 

    Preconditions:
        adj_2_freq_modded and adj_2_total_freq have the same keys.

    Arguments:
        adj_2freq_modded {{str: int}} 
        adj_2_total_freq {{str: int}} 
    """
    odds_modded = {}
    for adj in adj_2freq_modded:
        odds_modded[adj] = adj_2freq_modded[adj] / (adj_2_total_freq[adj] - adj_2freq_modded[adj])
    return odds_modded

def _calculate_sim_adj(adv_embedding, adjs_modded_word_embedding_pairs, adjs_odd_modded):
    sim_adj_mod = 0
    num_missing_odd_modded = 0
    for word_embedding_pair in adjs_modded_word_embedding_pairs.items():
        adj_modded = word_embedding_pair[0]
        if adj_modded not in adjs_odd_modded: 
            num_missing_odd_modded += 1
            continue
        embedding = word_embedding_pair[1]
        if type(embedding) == np.ndarray:
            sim_adj_mod += cos(adv_embedding, embedding) * adjs_odd_modded[adj_modded]
    return sim_adj_mod / (len(adjs_modded_word_embedding_pairs) - num_missing_odd_modded)

def _calculate_sim_adj_control(adv_embedding, adjs_modded_2_embeddings, adjs_2_odds_modded):
    sim_adj_mod = 0
    num_missing_odd_modded = 0
    for adj, embedding in adjs_modded_2_embeddings.items():
        adj_modded = adj
        if adj_modded not in adjs_2_odds_modded: 
            num_missing_odd_modded += 1
            continue
        if type(embedding) == np.ndarray:
            sim_adj_mod += cos(adv_embedding, embedding) * adjs_2_odds_modded[adj_modded]
    return sim_adj_mod / (len(adjs_modded_2_embeddings) - num_missing_odd_modded)

def get_adjs_modded_in_decs(adv_2dec_2adjs_modded, decs, adv):
    adjs_modded_per_dec = []
    for dec in decs:
        if adv not in adv_2dec_2adjs_modded:
            adjs_modded_per_dec.append(set([]))
        elif dec in adv_2dec_2adjs_modded[adv]:
            adjs_modded_per_dec.append(adv_2dec_2adjs_modded[adv][dec])
        else:
            adjs_modded_per_dec.append(set([]))
    return adjs_modded_per_dec

def compute_sim_adj(adv, control, bin_start_dec, intf_attd_date, 
                    adv_2dec_2adjs_modded, adj_2dec_2freq_modded, adj_to_freq):
    intf_decs = range(bin_start_dec, roundup(intf_attd_date), 10)

    set_adjs_modded_intf = reduce(lambda s1, s2: s1.union(s2), get_adjs_modded_in_decs(adv_2dec_2adjs_modded, intf_decs, intf))
    if intf in set_adjs_modded_intf:
        set_adjs_modded_intf.remove(intf) 

    ctrl_decs = range(bin_start_dec, bin_start_dec + 30, 10) 
    set_adjs_modded_ctrl = reduce(lambda s1, s2: s1.union(s2), get_adjs_modded_in_decs(adv_2dec_2adjs_modded, ctrl_decs, control))

    if len(set_adjs_modded_intf) == 0 or len(set_adjs_modded_ctrl) == 0:
        print(f"Intensifier or control modified no adjectives: {intf},{control}, ")
        return (Optional(0), Optional(0))

    intf_adj_modded_to_tp_freq_modded = compute_freq_modded(set_adjs_modded_intf, adj_2dec_2freq_modded, intf_decs) 
    ctrl_adj_modded_to_tp_freq_modded = compute_freq_modded(set_adjs_modded_ctrl, adj_2dec_2freq_modded, ctrl_decs)

    intf_adj_modded_to_tp_freq_total = compute_freq_total(set_adjs_modded_intf, adj_to_freq, intf_decs)
    ctrl_adj_modded_to_tp_freq_total = compute_freq_total(set_adjs_modded_ctrl, adj_to_freq, ctrl_decs)

    intf_adjs_odds_modded = compute_odds_modded(intf_adj_modded_to_tp_freq_modded, intf_adj_modded_to_tp_freq_total)
    ctrl_adjs_odds_modded = compute_odds_modded(ctrl_adj_modded_to_tp_freq_modded, ctrl_adj_modded_to_tp_freq_total)

    time_range = range(bin_start_dec, bin_start_dec + 30, 10)
    intf_source = get_closest_adjs([intf])[0].val
    ctrl_source = get_closest_adjs([control])[0].val

    intf_assoc_words = [intf_source] + list(set_adjs_modded_intf)
    intf_to_attd_date = {word: intf_attd_date for word in intf_assoc_words}
    intf_and_adjs_modded_embeddings = get_avg_hw_embeddings(time_range, intf_assoc_words, 
                                                 intf_to_attd_date) 

    intf_embedding = intf_and_adjs_modded_embeddings[0][1] 
    if type(intf_embedding) != np.ndarray: 
        print(f"Intensifier adjectival base has NAN embedding {intf_source}")
        return (Optional(), Optional())
        
    intf_adjs_modded_embedding_pairs = intf_and_adjs_modded_embeddings[1:]
    intf_sim_adj_mod = _calculate_sim_adj(intf_embedding, intf_adjs_modded_embedding_pairs, intf_adjs_odds_modded)


    ctrl_assoc_words = [ctrl_source] + list(set_adjs_modded_ctrl)
    ctrl_and_adjs_modded_embeddings = get_avg_hw_embeddings(time_range, ctrl_assoc_words, 
                                                 {}) 
    ctrl_embedding = ctrl_and_adjs_modded_embeddings[0][1] 
    if type(ctrl_embedding) != np.ndarray: 
        print(f"Control adjectival base has NAN embedding {ctrl_source}")
        return (Optional(), Optional())
    ctrl_adjs_modded_embedding_pairs = ctrl_and_adjs_modded_embeddings[1:]
    ctrl_sim_adj_mod = _calculate_sim_adj(ctrl_embedding, ctrl_adjs_modded_embedding_pairs, ctrl_adjs_odds_modded)

    return (Optional(intf_sim_adj_mod), Optional(ctrl_sim_adj_mod))

def compute_sim_adj_control(frame, adv_2dec_2adjs_modded, adj_2dec_2freq_modded, adj_to_freq):
    frame = frame.copy()
    frame['simadj'] = frame.apply(partial(compute_sim_adj_control_per_adv, adv_2dec_2adjs_modded, adj_2dec_2freq_modded, adj_to_freq), 
                                                    axis=1) 
    return frame                                                

def compute_sim_adj_control_per_adv(adv_2dec_2adjs_modded, adj_2dec_2freq_modded, adj_to_freq, row):
    adv, adj, data_decs, types = row.name, row.adjs, row.data_decs, row.type
    if types == 'ctrl':
        adv = adv[0: adv.index('_')]
    set_adjs_modded = reduce(lambda s1, s2: s1.union(s2), get_adjs_modded_in_decs(adv_2dec_2adjs_modded, data_decs, adv))
    if adv in set_adjs_modded:
        set_adjs_modded.remove(adv) 

    if len(set_adjs_modded) == 0: 
        print(f"Adverb {adv} modified no adjectives in the {data_decs}")
        return 0 
    
    adj_modded_to_tp_freq_modded = compute_freq_modded(set_adjs_modded, adj_2dec_2freq_modded, data_decs) 
    adj_modded_to_tp_freq_total = compute_freq_total(set_adjs_modded, adj_to_freq, data_decs)

    adjs_odds_modded = compute_odds_modded(adj_modded_to_tp_freq_modded, adj_modded_to_tp_freq_total)

    adv_and_adjs_modded = [adj] + list(set_adjs_modded)
    adv_and_adjs_modded_2_embeddings = load_avg_hw_embeddings_map(data_decs, adv_and_adjs_modded ) 

    if adj in adv_and_adjs_modded_2_embeddings:
        adv_embedding = adv_and_adjs_modded_2_embeddings[adj]
    else:
        print(f"Embedding for {adv} was not found in HistWords!")
        return np.nan
                                
    adjs_modded_2_embedding = {adj: adv_and_adjs_modded_2_embeddings[adj] for adj in set_adjs_modded if adj in adv_and_adjs_modded_2_embeddings}
    adv_sim_adj_mod = _calculate_sim_adj_control(adv_embedding, adjs_modded_2_embedding, adjs_odds_modded)
    return adv_sim_adj_mod
