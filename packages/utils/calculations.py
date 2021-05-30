from numpy import dot, sqrt
from math import exp
import numpy as np

def cos(vA,vB):
    """
    regular cosine similarity for two vectors 'vA' and 'vB'
    """
    denom = (sqrt(dot(vA,vA)) * sqrt(dot(vB,vB)))
    if denom == 0:
        print("Denom is 0, setting to 1")
        denom = 1
    return dot(vA, vB) / denom

def jaccard_similarity(iter1, iter2):
    first_set, second_set = set(iter1), set(iter2)
    return 1 - len(first_set.intersection(second_set))/len(first_set.union(second_set))

def normalize(vec):
    return vec/sqrt(sum(vec**2))

def sigmoid(x):
    return 1/(1 + exp(-x))