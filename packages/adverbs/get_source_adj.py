from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches as gcm
from ..utils.util_objects import Optional

from enum import Enum 


HARDCODED = {
    'astoundingly': 'astounding',
    'frightfully': 'frightful',
    'supposedly': 'supposed',
    'colossally': 'colossal', 
    'fabulously': 'fabulous', 
    'nauseatingly': 'nauseating', 
    'unmitigatedly': 'unmitigated', 
    'approximately': 'approximate',
    'monumentally': 'monumental', 
    'mainly': 'main', 
    'undoubtedly': 'undoubtedly',
    'breathtakingly': 'breathtaking',
    'wholeheartedly': 'wholehearted',
    'mindblowingly': 'mindblowing', 
    'mindbogglingly': 'mindboggling', 
    'hellishly': 'hellish',
    'safely': 'safe',
    'veritably': 'veritable',
    'emphatically': 'emphatic',
    'delectably': 'delectable',
    'majorly': 'major',
    'notably': 'notable',
    'dismayingly': 'dismaying',
    'notoriously': 'notorious',
    'basically': 'basic', 
    'formerly': 'former',
    'frustratingly': 'frustrating', 
    'hugely': 'huge',
    'inestimably': 'inestimable',
    'lately': 'late', 
    'widely': 'wide',
    'abysmally': 'abysmal',
    'definitely': 'definite',
    'horrifically': 'horrific',
    'accordingly': 'according',
    'correctly': 'correct',
    'readily': 'ready',
    'profusely': 'profuse',
    'decadently': 'decadent',
    'inexplicably': 'inexplicable',
    'purely': 'pure',
    'newly': 'new',
    'normally': 'normal',
    'unbearably': 'unbearable',
    'unsettlingly': 'unsettling',
    'exceedingly': 'exceeding',
    'mightily': 'mighty',
    'unnervingly': 'unnerving',
    'egregiously': 'egregious',
    'unapologetically': 'unapologetic',
    'secondly': 'second',
    'epically': 'epic',
    'pronouncedly': 'pronounced', 
    'sinfully': 'sinful',
    'thunderingly': 'thundering',
    'phenomenonally': 'phenomenonal',
    'brutally': 'brutal', 
    'devastatingly': 'devastating', 
    'fantastically': 'fantastic',
    'dizzyingly': 'dizzying', 
    'staggeringly': 'staggering', 
    'awesomely': 'awesome', 
    'firstly': 'first', 
    'nightmarishly': 'nightmarish'
}

def get_closest_adjs(advs, num_close_match=1, no_match_fn=None):
    closest_advs = []
    for adv in advs:
        possible_adj = []
        for synset in wn.synsets(adv):
            for lemmas in synset.lemmas():
                for pertainym in lemmas.pertainyms():
                    possible_adj.append(pertainym.name())
        closest_adj = list(set(gcm(adv, possible_adj, n=num_close_match)))
        if closest_adj == []:
            if adv in HARDCODED:
                closest_advs.append(Optional(HARDCODED[adv])) # 
            else:
                closest_advs.append(Optional()) # 
                no_match_fn(adv)
        else:
            assert len(closest_adj) == 1
            closest_advs.append(Optional(closest_adj[0]))
    return closest_advs

def get_verb_source_lemma(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, wn.VERB) for word in words]