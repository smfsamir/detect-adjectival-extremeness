from .w2v import get_target_embedding

class Word2VecEmbeddings:
    def __init__(self, w2v_model):
        vocab = list(w2v_model.wv.index2word)
        self.input_embeddings = {} # Is there a better way to store these
        self.output_embeddings = {}
        for word in vocab:
            self.input_embeddings[word] = w2v_model.wv[word]
            self.output_embeddings[word] = get_target_embedding(w2v_model, word)

        
    def get_input_embedding(self, word):
        return self.input_embeddings[word] 

    def __getitem__(self, word):
        """Returns the input layer embedding from the word2vec model for {word}
        """
        return self.get_input_embedding(word)

    def get_output_embedding(self, word):
        return self.output_embeddings[word] 