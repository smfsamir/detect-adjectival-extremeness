import pandas as pd
from nltk.tag import pos_tag
from collections import OrderedDict
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def load_keyed_vectors():
    model = KeyedVectors.load_word2vec_format('data/static/w2v/GoogleNews-vectors-negative300.bin', binary=True)
    return model

def get_vocab():
    model = load_keyed_vectors()
    return model.wv.vocab

def get_embeddings(words, flag_for_missing=set([])):
    model = KeyedVectors.load_word2vec_format('data/static/w2v/GoogleNews-vectors-negative300.bin', binary=True)
    
    if type(words) == np.ndarray or type(words) == list or type(words) == set:
        word_2_vecs = OrderedDict()
        for word in words:
            if word in model:
                word_2_vecs[word] = model[word]
            else:
                if word in flag_for_missing:
                    print(f"{word} not available in synchronic w2v")
    else:
        print(f"Unrecognized type: {type(words)}")
    return word_2_vecs

def get_most_similar_words(words, n=10):
    mappings = {}
    model = load_keyed_vectors()
    for w in words:
        if w in model:
            mappings[w] = model.similar_by_word(w, topn=n)
    return mappings

def get_most_similar_by_vector(vector, n=50):
    model = load_keyed_vectors()
    ws_sims = model.similar_by_vector(vector, n)
    return ws_sims

def get_n_similarity(wl1, wl2): # word lists 
    model = load_keyed_vectors()
    wl1 = list(filter( lambda x: x in model, wl1))
    wl2 = list(filter( lambda x: x in model, wl2))

    wl1_embeds = np.array([model[w] for w in wl1])
    wl2_embeds = np.array([model[w] for w in wl2])

    mat = cosine_similarity(wl1_embeds, wl2_embeds)
    return mat, wl1, wl2

def get_embed_frame(ws, w_to_embeds):
    """Returns a dataframe where the indices are the {ws}
    and the embeds are taken from the dictionary w_to_embeds.

    Args:
        ws ([type]): [description]
        w_to_embeds ([type]): [description]
    """
    embeds = []
    for w in ws:
        embeds.append(w_to_embeds[w])
    # TODO: normalize here.
    embeds = normalize(np.array(embeds))
    embed_frame = pd.DataFrame(data=embeds)
    embed_frame['word'] = ws
    embed_frame.set_index('word', inplace=True)
    return embed_frame