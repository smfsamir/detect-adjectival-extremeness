import pandas as pd
import numpy as np
from ..pkl_operations.pkl_io import *
from .constants import INTENSIFIERS
from functools import reduce

def load_intf_to_attd_date():
    """Returns a map of all intensifiers to the attested date where they gained
    a (degree) intensifying sense.
    """
    frame_intensifier_dates = pd.read_csv("data/static/adverbs/intensifier_dates.csv", dtype={"attested_year": int})
    frame_intensifier_dates = frame_intensifier_dates.set_index("intensifier")
    intensifier_to_attested_date = frame_intensifier_dates.to_dict('index')
    intensifier_to_attested_date = {intf: intensifier_to_attested_date[intf]['attested_year'] for intf 
        in intensifier_to_attested_date}
    return intensifier_to_attested_date

def load_all_intfs_with_attd_dates():
    frame_intensifier_dates = pd.read_csv("data/static/adverbs/intensifier_dates.csv", dtype={"attested_year": int})
    return set(frame_intensifier_dates['intensifier'].values)

def load_dependency_count_map(mm_dash_dd):
    """Returns DataFrame with |word|1800_counts_adj|1800_counts_verb|1810_counts_adj|...
    """
    def merge_adv_counts_all(arc_names):
        adv_dep_counts = [load_pkl_from_path(f"data/artifacts/adv_dep_pos_counts/2020-{mm_dash_dd}/{arc_name}") for arc_name in arc_names] 
        merged_adv_counts = reduce(merge_adv_counts_two, adv_dep_counts)
        return merged_adv_counts

    def merge_adv_counts_two(adv_counts_one, adv_counts_two):
        new_counts = {}
        for key, value in adv_counts_one.items():
            if key in adv_counts_two:
                new_counts[key] = [adv_counts_one[key][0] + adv_counts_two[key][0], adv_counts_one[key][1] + adv_counts_two[key][1]]
            else:
                new_counts[key] = adv_counts_one[key] 

        for key, value in adv_counts_two.items():
            if not key in new_counts:
                new_counts[key] = value
        return new_counts
    arcs_range = list(range(1,99))
    get_num = lambda x: f'0{x}' if x < 10 else f'{x}'
    adv_dep_counts = merge_adv_counts_all([f"arcs.{get_num(num)}-of-99_adv_counts" for num in arcs_range])
    return adv_dep_counts

def get_pos_counts(advs_list, get_adj, intensifier_to_attested_date, adv_dep_counts):
    decades = list(range(1800,2010,10))
    decadal_counts = {}
    pos_i = (0 if get_adj else 1)
    key_suffix = ("adj" if get_adj else "verb")
    for i in range(len(decades)):
        decade_counts = []
        decade = decades[i]
        for adv in advs_list:
            if adv in intensifier_to_attested_date: 
                if intensifier_to_attested_date[adv] > decade:
                    # if adv_counts[adv][0][i] + adv_counts[adv][1][i] > 10:
                    decade_counts.append(adv_dep_counts[adv][pos_i][i])
                    # else:
                    #     decade_counts.append(np.nan)
                else: 
                    decade_counts.append(np.nan)
            else:
                decade_counts.append(adv_dep_counts[adv][pos_i][i])
        decadal_counts[f"{decade}_counts_{key_suffix}"] = decade_counts
    return decadal_counts

def load_dependency_count_frame(mm_dash_dd):
    """Returns a DataFrame containing |words|type|1810_counts_adj|1810_counts_verb|...
    Words are all adverbs. Types denotes whether the adverb denotes an intensifier or 
    a control word. The index is set to 'words'

    This frame is linked to the attested dates which the intensifiers gained a degree intensifying sense.
    The entry is NaN for all count columns (i.e., not counted) if the decade of the count column is past the decade
    at which the intensifier gained an intensifying sense.  
    """
    adv_counts = load_dependency_count_map(mm_dash_dd)
    advs_list = list(adv_counts.keys())
    all_intensifiers_of_interest = INTENSIFIERS
    types = ["intensifier" if adv in all_intensifiers_of_interest else "control" 
            for adv in advs_list]
    # intf_to_attd_date = load_intf_to_attd_date()
    decades = list(range(1800,2010,10))
    adj_counts = get_pos_counts(advs_list, True, {}, adv_counts)
    verb_counts = get_pos_counts(advs_list, False, {}, adv_counts )
    adj_counts.update(verb_counts)
    all_counts = adj_counts
    frame_elements = {"words": advs_list, "types": types}
    # TODO: run a mann-whitney u test
    frame_elements.update(all_counts)
    frame = pd.DataFrame(frame_elements)
    return frame.set_index('words')

if __name__ == '__main__':
    print(load_all_intfs_with_attd_dates())