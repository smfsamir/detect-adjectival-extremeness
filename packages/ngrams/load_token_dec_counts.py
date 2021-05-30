from ..data_management.load_unigrams import get_counts_of_words
from ..pkl_operations.pkl_io import store_results_dynamic 
from ..adverbs.constants import *

def save_counts_per_dec(words, fname):
    """Returns frequencies of the tokens for every decade
    from 1800s to the 2000s

    Arguments:
        tokens {[str]} -- List of tokens 

    Returns 
        {{token: np.array(21)}} 
    """
    words = sorted(words)

    word_to_counts = {}
    i = 0
    start_i = 0
    words_len = len(words)
    while i != len(words):
        start_letter = words[i][0]
        while i < words_len and words[i][0] == start_letter:
            i += 1
        selected_advs = words[start_i:i]
        print(f"Processing {selected_advs}")
        start_i = i
        pass
        all_counts = get_counts_of_words(selected_advs)
        for j in range(0,len(all_counts)):
            word_to_counts[selected_advs[j]] = all_counts[j]
    store_results_dynamic(word_to_counts, fname, DYNAMIC_ARTIFACTS_PATH)