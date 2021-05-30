from collections import defaultdict
from ..adverbs.data_load import load_all_intfs_with_attd_dates
from ..diachronic_surprisal.trigram_write import load_ngrams, rm_ngrams
from ..adverbs.constants import INTENSIFIERS, CONTROL_WORDS
from ..adverbs.get_source_adj import get_closest_adjs
from ..pkl_operations.pkl_io import store_results_dynamic
from ..utils.util_functions import round_down
import numpy as np

zero_arr = lambda: np.zeros(21)
class ModifierAdjFreq:
    def __init__(self, modifier, bg_counts=None):
        self.modifier = modifier
        if bg_counts is None:
            self.bg_counts = defaultdict(zero_arr)
        else:
            self.bg_counts = bg_counts

    def update(self, word, dec, count):
        if dec >= 1800:
            self.bg_counts[word][round_down(dec - 1800)//10] += count
    
    def __repr__(self):
        return f"ModifierAdjFreq({self.modifier})"

class Bigram:
    def __init__(self, line):
        split_line = line.split('\t')
        self.words = split_line[0].split(' ')
        self.year = int(split_line[1])
        self.num_occurrences = int(split_line[2])
        self.last_word = self.words[-1]
        self.first_word = self.words[0]

BIGRAM_DIR = "data/static/ngrams/bigrams"
BIGRAM_FNAME_BASE = "googlebooks-eng-all-2gram-20120701"

def process_bigram_file(bigram_dir, bigram_fname, ws_of_intr, lines_read_func, mod_adv_freq):
    mod = mod_adv_freq.modifier
    with open(f'{bigram_dir}/{bigram_fname}.txt', 'r') as bigram_f:
        lines_read = 0
        for ngram_line in bigram_f:
            lines_read += 1
            bigram = Bigram(ngram_line)
            if bigram.first_word == mod:
                if bigram.last_word in ws_of_intr:
                    year = bigram.year
                    num_occurrences = bigram.num_occurrences

                    mod_adv_freq.update(bigram.last_word, year, num_occurrences)
            lines_read_func(lines_read)


def compute_bg_freq(modifier_adv_freq, woi):
    modifier = modifier_adv_freq.modifier
    f_two_chars = modifier[0:2]

    bigram_fname = f'{BIGRAM_FNAME_BASE}-{f_two_chars}'
    load_ngrams(bigram_fname, BIGRAM_DIR)

    def lines_read_f(num_lines):
        if num_lines % 50000000 == 0:
            print(num_lines)
    process_bigram_file(BIGRAM_DIR, bigram_fname, woi, lines_read_f, modifier_adv_freq)

    store_results_dynamic(dict(modifier_adv_freq.bg_counts), f'{modifier}_adv_counts', 'data/artifacts/')
    rm_ngrams(BIGRAM_DIR, bigram_fname)