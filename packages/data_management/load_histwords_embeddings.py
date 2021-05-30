from collections import defaultdict
from ..utils.util_functions import round_down, roundup
from ..pkl_operations.pkl_io import load_pkl_from_path
from ..adverbs.constants import Vec

from sklearn.neighbors import NearestNeighbors
import numpy as np

def load_histwords(time_period, words, path):
    assert time_period in range(1800, 2010, 10), "Invalid time period: {}".format(time_period)
    vocab = load_pkl_from_path(path + "/{}-vocab".format(time_period))
    vocab_set = set(vocab)
    indices = []
    embedding_existing_words = []
    for word in words:
        if word in vocab_set:
            indices.append(vocab.index(word))
            embedding_existing_words.append(word)
    
    embeddings = np.load(path + "/{}-w.npy".format(time_period))
    return embedding_existing_words, embeddings[indices]

# TODO: make path customizable
def load_histwords_map(time_period, words, path="/hal9000/datasets/wordembeddings/historical_sgns_all_english", flag_for_missing=set([])):
    """Same as load_histwords

    Args:
        time_period (int): decade in 1800 to 2000 (inclusive)
        words ([str]): list of desired word embeddings
        path (str, optional): [description]. 

    Returns:
        ({str: np.array}):  Array of a subset of words (which were found in histwords embeddings) to their
        respective embeddings.
    """
    vocab = load_pkl_from_path(path + "/{}-vocab".format(time_period))
    vocab_set = set(vocab)
    indices = []
    embedding_existing_words = []
    for word in words:
        if word in vocab_set:
            indices.append(vocab.index(word))
            embedding_existing_words.append(word)
    
    embeddings = np.load(path + "/{}-w.npy".format(time_period))
    des_embeddings = embeddings[indices]
    w_to_embed = {}
    for i in range(len(embedding_existing_words)):
        if sum(des_embeddings[i]) == 0:
            if embedding_existing_words[i] in flag_for_missing:
                print(f'{embedding_existing_words[i]} is a zero vector in histwords in the {time_period}s')
        else:
            w_to_embed[embedding_existing_words[i]] = des_embeddings[i]

    return w_to_embed

def load_avg_hw_embeddings_map(decs, words): 
    w_to_all_embeds = defaultdict(list)
    for tp in decs: # time period
        w_to_tp_embeds = load_histwords_map(tp, words)
        for word in w_to_tp_embeds:
            w_to_all_embeds[word].append(w_to_tp_embeds[word])
    w_to_mean_embeds = {}
    for w in w_to_all_embeds:
        if w_to_all_embeds[w] != []:
            w_to_mean_embeds[w] = np.mean(w_to_all_embeds[w], axis=0)
    return w_to_mean_embeds

def get_avg_hw_embeddings(time_range, words, intf_to_attd_date):
    w_to_embeds = defaultdict(list)
    for time_period in time_range:
        if time_period >= 2000:
            continue
        found_words, embeddings = load_histwords(time_period, words)
        for i in range(len(found_words)):
            word = found_words[i]
            if (word not in intf_to_attd_date) or (roundup(intf_to_attd_date[word]) > time_period):
                if np.sum(embeddings[i]) != 0:
                    w_to_embeds[word].append(embeddings[i])
    w_mean_embed = []
    for word in words:
        if w_to_embeds[word] == []:
            w_mean_embed.append((word, Vec.NAN))
        else:
            w_mean_embed.append((word, np.mean(np.array(w_to_embeds[word]), axis=0)))
    return w_mean_embed

def load_all_histwords_in_decade(time_period, path="/hal9000/datasets/wordembeddings/historical_sgns_all_english" ):
    assert time_period in range(1800, 2010, 10), "Invalid time period: {}".format(time_period)
    vocab = load_pkl_from_path(path + "/{}-vocab".format(time_period))
    embeddings = np.load(path + "/{}-w.npy".format(time_period))
    return vocab, embeddings

def get_nns_for_decade(time_period, embed, freq_cutoff):
    nn_struct = NearestNeighbors(metric='euclidean', n_neighbors=50, algorithm='brute')

    vocab, embeddings = load_all_histwords_in_decade(time_period)
    return vocab, nn_struct.fit(embeddings)