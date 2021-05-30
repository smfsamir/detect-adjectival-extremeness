"""Object for storing and reading from a (particularly large) corpus
such that the documents/sentences are generated one at a time instead of at once.
"""
class SingleDocumentCorpus:
    def __init__(self, file_path):
        self.file_path = file_path




class MultiDocumentCorpus: