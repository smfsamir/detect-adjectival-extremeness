from functools import partial
import pandas as pd
import numpy as np

from ..data_management.load_histwords_embeddings import load_histwords_map, load_avg_hw_embeddings_map
from ..adverbs.classification import single_centroid_classify_w_embeds_query_only

########################################### Below are methods for all decades
def single_centroid_prob_extreme(adv_frame, extreme_adjs, dec):
    query_2_embeds = load_histwords_map(dec, adv_frame.adjs)
    extreme_adj_2_embeds = load_histwords_map(dec, extreme_adjs)

    extremeness_val_frame = single_centroid_classify_w_embeds_query_only(extreme_adj_2_embeds, query_2_embeds, True)
    return extremeness_val_frame['sim_to_extr_centroid'].values

########################################### Below are methods for the HTE experiment

def centroid_prob_extreme_controlled(adv_frame, extreme_adjs):
    adv_frame = adv_frame.copy()
    adv_frame['prob_extreme'] = adv_frame.apply(partial(centroid_prob_extreme_controlled_per_row, extreme_adjs), 
                                                    axis=1)
    return adv_frame

def centroid_prob_extreme_controlled_per_row(extreme_adjs, row):
    query, data_decs = row.adjs, row.data_decs

    pos_seed_2embeds = load_avg_hw_embeddings_map(data_decs, extreme_adjs)

    query_2embeds = load_avg_hw_embeddings_map(data_decs, [query])
    if query_2embeds == {}:
        return np.nan
    exemplar_frame = single_centroid_classify_w_embeds_query_only(pos_seed_2embeds, 
                                    query_2embeds,
                                    True) 
    return exemplar_frame['sim_to_extr_centroid'].values[0]