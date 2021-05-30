class MeanStdErr:
    def __init__(self, mean, stderr):
        self.mean = mean 
        self.stderr = stderr

class MeanConfInterval():
    def __init__(self, mean, conf_interval):
        self.mean = mean
        self.conf_interval = conf_interval

class WordNumPair():
    def __init__(self, num, word):
        self.num = num
        self.word = word
    
class LineWithPos:
    def __init__(self, line_with_pos):
        self.line_with_pos = [TokenPOS(token_pos[0], token_pos[1]) for token_pos in line_with_pos]
    
    def __iter__(self):
        yield from self.line_with_pos
    
    def __len__(self):
        return len(self.line_with_pos)

    def __getitem__(self, i):
        return self.line_with_pos[i]

    def _get_words_only(self):
        return [token_pos.token for token_pos in self.line_with_pos]

    def __repr__(self):
        return " ".join(self._get_words_only())
    def get_intersecting_words(self, set_words):
        return set(self._get_words_only()).intersection(set_words)

class TokenPOS:
    def __init__(self, token, pos):
        self.token = token
        self.pos = pos
    
    def __repr__(self):
        return f"({self.token}, {self.pos})"
        
class Optional():
    def __init__(self, value=None): 
        if value:
            self.exists = True
            self.val = value 
        else:
            self.exists = False
            self.val = None

    def __repr__(self):
        return f"Optional({self.val})" if self.exists else "Optional()"