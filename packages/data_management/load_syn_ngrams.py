import math
import numpy as np
from collections import defaultdict 

from ..utils.util_functions import round_down
roundup = lambda x: int(math.floor(x/10))*10

class CountsByYear:
    def __init__(self, counts_by_year, decade_range=list(range(1800,2010,10))):
        decade_to_counts = defaultdict(int)
        for count_by_year in counts_by_year:
            year, count = count_by_year.split(',')
            decade = round_down(int(year))
            decade_to_counts[decade] += int(count)
        self.counts_by_year = decade_to_counts 

    def __getitem__(self, key):
        return self.counts_by_year[key]

    def __contains__(self, key):
        return key in self.counts_by_year 

    def get_decades(self):
        """Return the decades in which this ngram appears
        """
        return self.counts_by_year.keys()

    def get_dec_to_count_map(self):
        return self.counts_by_year



class SynToken:
    def __init__(self, ngram):
        if ngram[0] == '/':
            contents = ngram.split('/')[1:]
            if len(contents) != 4:
                print(contents)
                self.is_invalid = True
                word, pos_tag, dep_label, head_index = ['', '', '', '']
            else:
                word, pos_tag, dep_label, head_index = contents
        else:
            word, pos_tag, dep_label, head_index = ngram.split('/')[0:4]
        self.is_invalid = not str.isnumeric(head_index)
        self.word = word
        self.pos_tag = pos_tag
        self.dep_label = dep_label
        self.head_index = head_index


    def __str__(self):
        return f"{self.word}/{self.pos_tag}"

class SynNgram:
    def __init__(self, ngram_line):
        ngram_line = ngram_line.strip()
        ngram_tab_split = ngram_line.split('\t')
        head, tokens, total_count = ngram_tab_split[0:3]
        counts_by_year = ngram_tab_split[3:]
        self.head = head
        self.total_count = total_count
        self.counts_by_year = CountsByYear(counts_by_year)
        self.tokens = [SynToken(token) for token in tokens.split(' ') ]
        self.is_invalid = any([token.is_invalid for token in self.tokens])
    
    def __str__(self):
        return str(self.tokens)
    
    def __contains__(self, word):
        return any([word == token.word for token in self.tokens])
    
    def __getitem__(self, token_i):
        """Get the token at position {token_i} in the 
        syn_ngram.

        Arguments:
            token_i {int} -- from 0 to len of the ngram
        
        Returns:
            The word for the token at position i
        """
        return self.tokens[token_i]
    
    def inverse_contains(self, words):
        """Check if any of the tokens in this ngram are in {words}

        Arguments:
            words {set} -- Set of words

        Returns:
            True|False
        """
        return any([token.word in words for token in self.tokens]) 

    def get_tag_of_dep(self, adv): 
        other_token_tag = None
        for token in self.tokens:
            if token.word != adv:
                other_token_tag = token.pos_tag
        return other_token_tag
    
    def get_200_year_counts(self):
        """Return a list of the counts from the year 1800 (inclusive)
        to 2000 (inclusive).
        """
        year_range = (range(1800, 2010, 10))
        return np.array([self.counts_by_year[year] 
            if year in self.counts_by_year else 0
            for year in year_range])

def load_ngrams_syn(syn_fname):
    with open(f"data/static/syn_ngrams/arcs/{syn_fname}", 'r') as syn_ngrams:
        for line in syn_ngrams:
            yield SynNgram(line)