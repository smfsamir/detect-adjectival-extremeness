from ..utils.util_objects import LineWithPos
from ..utils.util_functions import copy_count_dict, convert_nested_defaultdict_to_dict, compute_npmi
from ..pkl_operations.pkl_io import store_results_dynamic
from ..adverbs.constants import DYNAMIC_ARTIFACTS_PATH

from collections import defaultdict
from scipy.stats import fisher_exact
import os
from chardet import detect
import re
import numpy as np

from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, MLEProbDist
from nltk.corpus import stopwords
from nltk import bigrams, trigrams


def load_generator_coca_corpus(ignore_stopwords, coca_path="data/static/COCA/corpus_files/"):
    """Read lines from the COCA corpus one at a time.
    """
    eos_regex = re.compile(r"(!|\.|\?)")
    ignore_regex = re.compile(r"@")
    later_re = r"\d+(_\w+)+\.txt"
    
    if ignore_stopwords:
        stop_ws = set(stopwords.words('english'))
        ignore = stop_ws.union(set(["#"]))
    else:
        ignore = set([])

    for filename in os.listdir(coca_path):
        if ".txt" not in filename:
            continue
        with open(coca_path+filename, 'r', encoding='ascii', errors="ignore") as coca_file:
            sentence = []
            for line in coca_file:
                split_line = line.strip().split('\t')
                if re.match(later_re, filename):
                    if len(split_line) < 5:  
                        continue
                    token = split_line[2].lower()
                    pos_tag = split_line[4].strip()
                else:
                    if len(split_line) < 3:  
                        continue
                    token = split_line[0].lower()
                    pos_tag = split_line[2].strip()
                
                if not re.match(ignore_regex, token) and token not in ignore:   
                    sentence.append((token, pos_tag))
                if re.match(eos_regex, token):
                    sentence = LineWithPos(sentence)
                    yield sentence 
                    sentence = []

def get_freq_unigrams(tokens_to_count):
    w_to_freq = defaultdict(int)
    n_s_read = 0
    total_freq = 0
    for sentence in load_generator_coca_corpus(False):
        n_s_read +=1
        for token_pos in sentence: 
            total_freq += 1
            if token_pos.token in tokens_to_count:
                w_to_freq[token_pos.token] += 1
        if n_s_read % 1000000==0:
            print(f"{n_s_read} sentences read!")
    print(f"There were {total_freq} tokens read!")
    return w_to_freq

def compute_pmi_deps_coca(edms, adjectives, scms, fname_prefix):
    """Compute the PMI of adjectives in {adjectives} with extreme degree modifiers in 
    {edms} and scalar modifiers in {scms}.   

    Arguments:
        edms {set([str])}
        adjectives {set([str])} 
        edms {set([str])}
        adjectives {set([str])} 

    Returns: 
        [(adjective, pmi_with_edm, pmi_with_very)]
    """
    freq_dms_and_adjs = defaultdict(int)
    mods_2adjs_2counts = defaultdict(lambda: defaultdict(int)) 
    total = 0
    num_s_read = 0
    for sentence in load_generator_coca_corpus(False):
        sen_len = len(sentence)
        for i in range(sen_len):
            total += 1
            token_at_i = sentence[i].token
            if token_at_i in edms or token_at_i in scms:
                if (i+1 < sen_len) and (sentence[i+1].token in adjectives):
                    mods_2adjs_2counts[token_at_i][sentence[i+1].token] += 1
                freq_dms_and_adjs[token_at_i] += 1
            elif token_at_i in adjectives:
                freq_dms_and_adjs[token_at_i] += 1
        num_s_read += 1
        if num_s_read % 1000000 == 0:
            print(f"Read {num_s_read} lines!")
    print(f"Total number of tokens: {total}")
    store_results_dynamic(copy_count_dict(freq_dms_and_adjs), f"{fname_prefix}_coca_freq_dms_and_adjs", "data/artifacts")
    store_results_dynamic(convert_nested_defaultdict_to_dict(mods_2adjs_2counts), f"{fname_prefix}_coca_dms_2adjs_2counts", "data/artifacts")

def compute_pmi_coca(edms, adjectives, scms, mods_2adjs_2counts, freq_dms_and_adjs, vocab_size, use_smoothing=False ):
    adj_2jfreq_w_edms = defaultdict(int)
    adj_2jfreq_w_scms = defaultdict(int)
    for mod in mods_2adjs_2counts:
        if mod in edms:
            for adj, count in mods_2adjs_2counts[mod].items():
                adj_2jfreq_w_edms[adj] += count
        elif mod in scms:
            for adj, count in mods_2adjs_2counts[mod].items():
                adj_2jfreq_w_scms[adj] += count
    
    pmi_triples = []
    freq_all_edms = sum([freq_dms_and_adjs[edm] for edm in edms])
    freq_all_scms = sum([freq_dms_and_adjs[scm] for scm in scms])
    total = 951144190 
    for adj in adjectives:
        npmi_w_edm = compute_npmi('edm', adj, freq_all_edms, freq_dms_and_adjs[adj], adj_2jfreq_w_edms[adj], total, vocab_size)
        npmi_w_scm = compute_npmi('scm', adj, freq_all_scms, freq_dms_and_adjs[adj], adj_2jfreq_w_scms[adj], total, vocab_size)

        pmi_triples.append((adj, npmi_w_scm, npmi_w_edm))

    store_results_dynamic(pmi_triples, "coca_pmi_triples", "data/artifacts")
    return pmi_triples