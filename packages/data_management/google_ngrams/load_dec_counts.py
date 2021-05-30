## I can then put this into a table
from collections import defaultdict
from packages.pkl_operations.pkl_io import store_results_dynamic
from packages.utils.util_functions import round_down

def map_dec_to_num_tokens():
    dec_to_num_tokens = defaultdict(int)
    with open('data/static/ngrams/unigram/googlebooks-eng-all-totalcounts-20120701.txt') as tot_counts_f: 
        dec_counts_line = tot_counts_f.readline()
        dec_counts = dec_counts_line.split('\t')
        for dec_count in dec_counts:
            if dec_count == ' ' or dec_count == '':
                continue
            count_elems = dec_count.split(',') 
            decade = round_down(int(count_elems[0]))
            if decade < 1800: 
                continue
            count = int(count_elems[1])
            dec_to_num_tokens[decade] += count
    return dec_to_num_tokens

def map_adj_to_freq(adjs, ):