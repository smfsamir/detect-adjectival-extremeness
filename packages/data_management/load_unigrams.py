import numpy as np
class Unigram:
    def __init__(self, line):
        split_line = line.split('\t')
        self.word = split_line[0]
        self.year = int(split_line[1])
        self.num_occurrences = int(split_line[2])

def check_and_print(num_lines_read):
    if num_lines_read % 1000000 == 0:
        print(f"{num_lines_read} lines read!")

def get_counts_of_words(words):
    """Return a vector of size 21 which contains the frequency of all words in {words}
    from every decade in the time range 1800s-2000s.

    Arguments:
        words {[type]} -- [description]
    """
    path = 'data/static/ngrams/unigram/'
    unigram_file = f"googlebooks-eng-all-1gram-20120701-{words[0][0]}"
    w2i = {words[i]: i for i in range(len(words))}
    num_lines_read = 0
    set_words = set(words)
    instances = [] # going to maintain a list of triples (word, year, num_occurrences)
    count_array = np.zeros((len(words), 21))
    with open(path + unigram_file) as o_uni_f:
        for line in o_uni_f:
            num_lines_read += 1
            check_and_print(num_lines_read)
            unigram = Unigram(line)
            if unigram.word in set_words and unigram.year >= 1800:
                instances.append((unigram.word, unigram.year, unigram.num_occurrences))
    for instance in instances:
        count_array_i = w2i[instance[0]]
        year_i = (instance[1] - 1800) // 10
        count_array[count_array_i][year_i] += instance[2]
    print(count_array)
    return count_array