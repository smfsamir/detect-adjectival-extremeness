import os
import re 
import pandas as pd

PATH_PREFIX = 'data/static/intensity_ratings'
PATH_SUFFIX = 'gold_rankings'

dsets = ['demelo', 'crowd', 'wilkinson']

def is_ranking_file(fname):
    return 'rankings' in fname

def proc_line(line):
    elems = line.split('\t')
    adj_content = elems[1].strip()
    if '||' in adj_content:
        multi_adjs = adj_content.split('||')
        return list(map(lambda x: x.strip(), multi_adjs))
    else: 
        return [adj_content] # list of adjectives in line  

def rev_ranks(ranks):
    last_rank = ranks[0]
    reversed_ranks = [1 + last_rank - x for i, x in enumerate(ranks)]
    return reversed_ranks

def read_ranking_file(path, fname):
    with open(f'{path}/{fname}', 'r') as rank_f:
        adjs = [] # ordered by increasing intensity
        ranks = [] # required due to potential ties
        cur_rank = 1
        for line in rank_f:
            line_adjs = proc_line(line)
            adjs.extend(line_adjs)
            ranks.extend([cur_rank] * len(line_adjs))
            cur_rank += 1
        scale_name = fname[0: fname.index('.')]
        scale_names = [scale_name] * len(adjs)
        frame = pd.DataFrame({'adj': list(reversed(adjs)), 
                              'scale': scale_names, 
                              'ranks': rev_ranks(list(reversed(ranks)))})
        return frame

def load_scales():
    dset_frames = []
    for dset in dsets:
        full_path = f"{PATH_PREFIX}/{dset}/{PATH_SUFFIX}"
        curr_frames = []
        for fname in os.listdir(full_path):
            if is_ranking_file(fname):
                frame = read_ranking_file(full_path, fname)
                curr_frames.append(frame)
        dset_frames.append(pd.concat(curr_frames))
    return pd.concat(dset_frames)