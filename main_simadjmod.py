"""This module contains code for obtaining the unigram and bigram frequencies
required to compute the SimAdjMod feature. Specifically, it contains code for obtaining:

(1) Map of adverbs to decade to the adjectives that the adverb modified in that decade. 
Determined from Google Syntactic N-grams.

(2) Map of adjectives to decades to the number of times the adjectives 
were modified by an adverb for that decade. Determined from Google Syntactic N-grams.

(3) Map of adjectives to decades to the frequency of the adjectives for the decade. 
Determined from Google N-grams.
"""
import argparse
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from functools import reduce

from packages.data_management.load_syn_ngrams import load_ngrams_syn
from packages.pkl_operations.pkl_io import *
from packages.adverbs.constants import *
from packages.ngrams.load_token_dec_counts import save_counts_per_dec
from packages.adverbs.compute_sim_adj import compute_sim_adj

def lambda_adv_to_adj_per_dec_init():
    return defaultdict(set)

def lambda_adj_to_count_per_dec_init():
    return defaultdict(int)

def copy_count_dict(count_dict):
    return {k: v for k, v in count_dict.items()}

def convert_to_dict(adv_to_adjs_modded):
    for adv in adv_to_adjs_modded:
        adv_to_adjs_modded[adv] = dict(adv_to_adjs_modded[adv])
    return dict(adv_to_adjs_modded)

def load_advs(fname):
    advs = []
    path = "data/static/adverbs/"
    with open(path+fname+".txt", 'r')  as f:
        advs.extend([line.strip() for line in f.readlines()])
    return advs

def compute_adv_to_adj_for_file(arc_fname):
    """Compute data structure of the form 

    {adv: {1800: {}, ..., }, ...

    Arguments:
        arc_fname {[type]} -- [description]
    """
    INTENSIFIERS = load_advs("intensifiers_2")
    CONTROL_WORDS = load_advs("control_verbs")
    control_set = set(CONTROL_WORDS)
    all_intensifiers_of_interest = set(INTENSIFIERS)
    all_advs = control_set.union(all_intensifiers_of_interest)

    adv_to_adjs_modded = defaultdict(lambda_adv_to_adj_per_dec_init)

    adj_tag_re = re.compile(r'JJ')
    word_re = re.compile(r'[a-zA-Z]+')
    ngram_count = 0
    for syn_ngram in load_ngrams_syn(arc_fname):
        ngram_count += 1
        if syn_ngram.inverse_contains(all_advs) and not syn_ngram.is_invalid:
            for i in range(len(syn_ngram.tokens)):
                if syn_ngram[i].word in all_advs: 
                    adv = syn_ngram[i].word
                    head_index = int(syn_ngram[i].head_index )
                    head_token = syn_ngram[head_index - 1]


            if re.match(adj_tag_re, head_token.pos_tag): 
                if not re.match(word_re, head_token.word):
                    continue
                decades = syn_ngram.counts_by_year.get_decades() 
                for decade in decades:
                    adv_to_adjs_modded[adv][decade].add(head_token.word)
        if ngram_count % 1000000 == 0:
            print(f"Read {ngram_count} ngrams")
    std_dict = convert_to_dict(adv_to_adjs_modded)
    store_results_dynamic(std_dict, f"{arc_fname}_adv_to_adj_map", "data/artifacts")


def get_adj_to_freq_modified(adjs, arc_fname):
    """
    Return a mapping of {adj: {1800: num_modified_adv, 1810: num_modified_adv}}

    Arguments:
        arc_fname {str} -- Filename of a syntactic ngram arc file
    """
    adv_tag_re = re.compile(r'RB')
    adj_tag_re = re.compile(r'JJ')
    adv_to_adjs_modded = defaultdict(lambda_adj_to_count_per_dec_init)

    for syn_ngram in load_ngrams_syn(arc_fname):
        if syn_ngram.inverse_contains(adjs) and not syn_ngram.is_invalid:
            for i in range(len(syn_ngram.tokens)):
                if syn_ngram[i].word in adjs and re.match(adj_tag_re, syn_ngram[i].pos_tag) : 
                    adj = syn_ngram[i].word
                    for i in range(len(syn_ngram.tokens)):
                        if re.match(adv_tag_re, syn_ngram[i].pos_tag):
                            for decade, count in syn_ngram.counts_by_year.get_dec_to_count_map().items():
                                adv_to_adjs_modded[adj][decade] += count
                    break

    std_dict = convert_to_dict(adv_to_adjs_modded)
    store_results_dynamic(std_dict, f"{arc_fname}_adj_2_dec_to_advmodcount_map", "data/artifacts")

def obtain_all_adjs(adv_2_dec_2_adj):
    """Given a mapping of {adv: {dec: {adj1, adj2,...}}}, collapse
    all of the adjs into a single set 

    Arguments:
        adv_2_dec_2_adj {[type]} -- [description]
    """
    all_modded_adjs = set([])
    for adv, dec_2_adjs in adv_2_dec_2_adj.items():
        for dec, set_adjs in dec_2_adjs.items():
            all_modded_adjs.update(set_adjs)
    return all_modded_adjs

def collate_2_nested_dicts(word_2dec_2prop_one,  word_2dec_2prop_two, combine_fn):
    collated = {}
    def combine_2_dec_2props(dec_2prop_one, dec_2prop_two):
        combined_counts = {}
        for dec, adv_mod_count_one in dec_2prop_one.items():
            if dec in dec_2prop_two:
                combined_counts[dec] = combine_fn(adv_mod_count_one, dec_2prop_two[dec])
            else:
                combined_counts[dec] = adv_mod_count_one

        for dec, adv_mod_count_two in dec_2prop_two.items():
            if dec not in combined_counts:
                combined_counts[dec] = adv_mod_count_two
        return combined_counts

    for adj, dec_2advmodcount in word_2dec_2prop_one.items():
        if adj in word_2dec_2prop_two: 
            collated[adj] = combine_2_dec_2props(dec_2advmodcount, word_2dec_2prop_two[adj])
        else:
            collated[adj] = dec_2advmodcount
    for adj, dec_2advmodcount in word_2dec_2prop_two.items():
        if adj not in collated:
            collated[adj] = dec_2advmodcount
    return collated

def get_fnames_for_arcs(fname_suffix):
    fnames = []
    for arc_num in range(0, 99):
        if arc_num < 10:
            arc_str = f"0{arc_num}"
        else:
            arc_str = str(arc_num)
        fnames.append(f"data/artifacts/2020-06-10/arcs.{arc_str}-of-99_{fname_suffix}")
    return fnames

get_fname = lambda num: f"arcs.0{num}-of-99" if num < 10 else f"arcs.{num}-of-99"

def main(args):
    if args.get_adv_to_adj:
        arc_fnames = [get_fname(num) for num in range(0,99)]
        for arc_fname in arc_fnames:
            print(f"Processing {arc_fname}")
            compute_adv_to_adj_for_file(arc_fname)
    elif args.get_adj_to_freq_modified:
        arc_fnames = [get_fname(num) for num in range(0,99)]
        for arc_fname in arc_fnames:
            print(f"Processing {arc_fname}")
            adv_2_dec_2_adjs = load_pkl_from_path(f"data/artifacts/2020-07-05/{arc_fname}_adv_to_adj_map") 
            all_modded_adjs = obtain_all_adjs(adv_2_dec_2_adjs)
            get_adj_to_freq_modified(all_modded_adjs, arc_fname)
    elif args.collate_adv_2_dec_2adjmod:
        fnames = get_fnames_for_arcs("adv_to_adj_map")
        combine_fn = lambda adj_modded_1, adj_modded_2: adj_modded_1.union(adj_modded_2)
        nested_dicts = (load_pkl_from_path(fname) for fname in fnames)
        collated = reduce(lambda dict1, dict2: collate_2_nested_dicts(dict1, dict2, combine_fn), nested_dicts)
        store_results_dynamic(collated, "collated_all_adjs_modified", 'data/artifacts')
    elif args.collate_all_adj_mod_counts:
        fnames = get_fnames_for_arcs("adj_2_dec_to_advmodcount_map")
        combine_fn = lambda count1, count2: count1 + count2
        nested_dicts = (load_pkl_from_path(fname) for fname in fnames)
        collated = reduce(lambda dict1, dict2: collate_2_nested_dicts(dict1, dict2, combine_fn), nested_dicts)
        store_results_dynamic(collated, "collated_adj_mod_counts", 'data/artifacts')
    elif args.collect_freqs_all_adjs_modded:
        advs_2dec_2adjs_modded = load_pkl_from_path('data/artifacts/2020-06-11/collated_all_adjs_modified')
        all_adjs = set([])
        for adv, dec_2_adjsmodded in  advs_2dec_2adjs_modded.items():
            for dec, adjs_modded in dec_2_adjsmodded.items():
                all_adjs.update(adjs_modded)
        save_counts_per_dec(list(all_adjs), 'adjs_modded_2total_freq')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--get_adv_to_adj', action='store_true')
    parser.add_argument('--get_adj_to_freq_modified', action='store_true')
    parser.add_argument('--collate_all_adj_mod_counts', action='store_true')
    parser.add_argument('--collate_adv_2_dec_2adjmod', action='store_true')
    parser.add_argument('--collect_freqs_all_adjs_modded', action='store_true')
    main(parser.parse_args())