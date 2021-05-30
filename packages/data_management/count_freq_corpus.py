from collections import Counter

def freq_map(corpus):
    """Return a count of all of the tokens in the corpus.

    Params:
        corpus {[str]} -- A list of tokens representing the corpus
    """
    counter = Counter()
    for token in corpus:
        counter[token] += 1
    return counter
