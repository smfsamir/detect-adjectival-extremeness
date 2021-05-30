import numpy as np
import math 
from datetime import datetime

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def round_down(x):
    return int(math.floor(x / 10.0)) * 10

def conv_table_to_latex_string(frame, columns, modify_columns={}):
    def print_fn(row):
        table_str = ""
        for column in columns:
            if column in modify_columns:
                str_val = modify_columns[column](row[column])
            else:
                str_val = str(row[column])
            table_str = table_str + str_val + " & "
        table_str = table_str[0: -2] + "\\\\"  
        print(table_str)
    frame.apply(print_fn, axis=1)

def copy_count_dict(count_dict):
    return {k: v for k, v in count_dict.items()}

def convert_nested_defaultdict_to_dict(key_2key_2v):
    for key in key_2key_2v:
        key_2key_2v[key] = dict(key_2key_2v[key])
    return dict(key_2key_2v)

def get_ranking(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def unzip_get_elem(array, index):
    """
    array {[(...)]} -- An array of tuples 
    """
    return list(zip(*array))[index]
    
def map_list(func, arr):
    return list(map(func, arr))

def map_unpack_opt(arr):
    return map(lambda x: x.val, arr)

def extend_pr_curve(long_r_prs, short_r_prs):
    long_i = 0
    short_i = 0
    extended_r_prs = [] 
    while long_i != len(long_r_prs):
        long_r_val = long_r_prs[long_i][0] # 0th index is for the recall value
        short_r_val = short_r_prs[short_i][0] # 0th index is for the recall value
        short_p_val = short_r_prs[short_i][1] # 0th index is for the recall value
        if long_r_val == short_r_val:
            extended_r_prs.append((short_r_val, short_p_val))
            short_i += 1
        elif long_r_val > short_r_val:
            extended_r_prs.append((long_r_val, short_p_val))
        elif long_r_val < short_r_val:
            extended_r_prs.append((long_r_val, short_p_val))
        long_i += 1
    return np.array(extended_r_prs)

def compute_npmi(w1, w2, w1_freq, w2_freq, bigram_freq, total_freq, vocab_size, use_smoothing=True):
    cond_prob = (bigram_freq + 1) / (w1_freq + vocab_size)
    smoothed_total = total_freq + (vocab_size * vocab_size)
    w2_prob  = (w2_freq + vocab_size) / smoothed_total 
    w1_prob = (w1_freq + vocab_size) / smoothed_total  

    pmi = np.log(cond_prob) - np.log(w2_prob)
    log_joint_prob = np.log(cond_prob) + np.log(w1_prob)
    return pmi/log_joint_prob

def get_artifacts_path(filename):
    date = datetime.today().strftime('%Y-%m-%d')
    dest = "data/artifacts/{}/{}".format( date,filename)
    return dest 

def get_today_date_formatted():
    date = datetime.today().strftime('%Y-%m-%d')
    return date

def list_diff(items1, items2):
    """Return all items that are in {items1} but not {items2}

    Args:
        items1 (list) 
        items2 (list) 
    """
    items_diff = []
    items_2_set = set(items2)
    for item in items1:
        if item not in items_2_set:
            items_diff.append(item)
    return items_diff