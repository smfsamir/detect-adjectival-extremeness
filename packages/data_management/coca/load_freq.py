import pandas as pd
import numpy as np

def load_by_freq(path='data/static/coca_freq/lemmas_60k_m1340.txt', use_lemma_index=False, desired_words=None):
    # only load adjectives. 
    # TODO: ne
    if use_lemma_index:
        frame = pd.read_csv(path, sep="\t", index_col='lemma')
    else:
        frame = pd.read_csv(path, sep="\t")

    if desired_words is not None:
        return frame.loc[frame['lemma'].isin(set(desired_words))]
    else:
        return frame

def is_frequently_capitalized(freq_frame, lemma):
    if freq_frame.loc[lemma]["%caps"] >= 0.4:
        return True

def get_freq_matched_adjs(adjs, exclusions = set([]), path='data/static/coca_freq/lemmas_60k_m1340.txt'):
    frame = load_by_freq()
    # then, just sort by frequency and get corresponding adjectives :)
    frame = frame.loc[frame['PoS'] == 'j']
    frame.set_index('lemma', inplace=True)
    frame = frame.sort_values('perMil')

    adj_to_match = {}
    all_adj_lemmas = frame.index.values
    set_adjs = set(adjs)
    matched = set([])
    def cannot_match(match_cand_adj):
        return match_cand_adj in exclusions or match_cand_adj in set_adjs or \
            '-' in match_cand_adj or match_cand_adj[-3:] =='ing' or \
                match_cand_adj[-4:] == 'less' or match_cand_adj in matched or \
                    'er' in match_cand_adj

    def get_step():
        rand = np.random.rand()
        return 1 if rand > 0.5 else -1

    def increment(step):
        if step < 0:
            return step - 1
        else:
            return step + 1


    for i in range(len(all_adj_lemmas)):
        if all_adj_lemmas[i] in set_adjs:
            curr_adj_to_match = all_adj_lemmas[i]
            j = get_step()
            while cannot_match(all_adj_lemmas[i + j]) or is_frequently_capitalized(frame, all_adj_lemmas[i+j]): 
                j = increment(j)
            adj_to_match_freq_pm = frame.loc[curr_adj_to_match].perMil
            adj_to_match[(curr_adj_to_match,adj_to_match_freq_pm)] = (all_adj_lemmas[i+j], frame.loc[all_adj_lemmas[i+j]].perMil)
            matched.add(all_adj_lemmas[i+j])
    return adj_to_match 

def get_total_freq(path='data/static/coca_freq/words_219k_m1340.txt'):
    frame = pd.read_csv(path, sep="\t", encoding='cp1252', error_bad_lines=False)
    total_freq = frame['freq'].sum()
    print(total_freq)
    print(frame)

def load_lemma_freqs_by_word_tag(lemmas):
    word_freq_frame = load_by_freq(desired_words=lemmas)
    return word_freq_frame[['PoS', 'perMil', 'freq', 'lemma']]


if __name__ == '__main__':
    total_freq = load_by_freq()
    print(total_freq)