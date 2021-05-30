import numpy as np

from ..data_management.load_histwords_embeddings import load_histwords_map
from ..adverbs.get_source_adj import get_closest_adjs
from ..utils.calculations import cos

def get_odds_modded(adj, adj_freq_moddeds, adj_freqs, dec):
    if adj not in adj_freq_moddeds:
        return 0
    adj_freq_modded = adj_freq_moddeds[adj]
    if dec not in adj_freq_modded:
        return 0
    else: 
        adj_freq_modded_dec = adj_freq_modded[dec]
        adj_total_freq = adj_freqs[adj][(dec - 1800)//10]
        return (adj_freq_modded_dec) / (adj_total_freq - adj_freq_modded_dec)

def compute_simadj_per_dec(advs, adv_adjs_modded, freq_adj, adj_freq_modded, dec, use_adjs=False):
    sim_adjs = []
    if not use_adjs:
        adv_to_adv_vecs = load_histwords_map(dec, advs) # load from histwords
    else:
        adj_bases = [opt.val for opt in get_closest_adjs(advs)]
        adv_to_adj_base = {advs[i]: adj_bases[i] for i in range(len(advs))}
        adj_to_adj_vecs = load_histwords_map(dec, adj_bases) # load from histwords
        adv_to_adv_vecs = {}
        for adv in advs: 
            adj_base = adv_to_adj_base[adv]
            if adj_base in adj_to_adj_vecs:
                adv_to_adv_vecs[adv] = adj_to_adj_vecs[adj_base]


    for adv in advs:
        if adv not in adv_to_adv_vecs:
            sim_adjs.append(np.nan)
            print(f"Adv {adv} not available in HW for the {dec}s")
            continue
        if np.sum(adv_to_adv_vecs[adv]) == 0:
            sim_adjs.append(np.nan)
            print(f"Adv {adv} is a zero vector in HW for the {dec}s")
            continue
        adv_vec = adv_to_adv_vecs[adv]
        if adv not in adv_adjs_modded:
            sim_adjs.append(np.nan)
            print(f"Adv {adv} did not modify any adjectives")
            continue
        if dec not in adv_adjs_modded[adv]:
            sim_adjs.append(np.nan)
            print(f"Adv {adv} did not modify any adjectives in the {dec}s")
            continue
        adjs_modded = adv_adjs_modded[adv][dec] 
        adj_to_adj_vecs = load_histwords_map(dec, adjs_modded)
        sim_adj_unnormed = 0
        num_adjs_counted = 0
        for adj, adj_vec in adj_to_adj_vecs.items():
            if np.sum(adj_vec) == 0:
                print(f"Adjective modified vector {adj} is zero in HW for the {dec}s")
                continue
            odds_modded = get_odds_modded(adj, adj_freq_modded, freq_adj, dec)
            if odds_modded == 0:
                print(f"Adjective modified vector {adj} has 0 modded probability for the {dec}s")                 
                continue
            num_adjs_counted += 1
            sim_adj_unnormed += cos(adv_vec, adj_vec) * odds_modded
        if num_adjs_counted == 0:
            sim_adjs.append(np.nan)
        else:
            sim_adjs.append(sim_adj_unnormed / num_adjs_counted)
    return sim_adjs