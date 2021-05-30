from ..pkl_operations.pkl_io import load_pkl_from_path
import numpy as np

def load_control_embeddings(time_period, words, path):
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