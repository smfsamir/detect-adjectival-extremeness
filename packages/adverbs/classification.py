from functools import reduce 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

from packages.utils.calculations import cos
from packages.synchronic_embeddings.synchronic_w2v import get_embeddings
from packages.pkl_operations.pkl_io import *

def single_centroid_classify_w_embeds_query_only(pos_seed_2embeds, queries_2_embeds, normalize_vecs):
    def get_vecs(adjs_2_embds):
        vecs = list(adjs_2_embds.values())
        if normalize_vecs:
            return normalize(vecs)
        else:
            return vecs

    pos_seed_embeds = get_vecs(pos_seed_2embeds)

    query_embeds = get_vecs(queries_2_embeds)

    extreme_centroid = pos_seed_embeds.mean(axis=0)
    extr_cent_sims = [cos(query_embed, extreme_centroid) for query_embed in query_embeds]

    frame = pd.DataFrame({
        'query': list(queries_2_embeds.keys()),
        'sim_to_extr_centroid': extr_cent_sims
    })

    return frame 