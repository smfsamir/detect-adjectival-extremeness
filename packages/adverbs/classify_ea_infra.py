import pandas as pd
import numpy as np

def build_pmi_frame(pmi_triples):
    words, pmi_very, pmi_edm = zip(*pmi_triples)

    frame = pd.DataFrame({
        'word': words, 
        'pmi_very': [pmi for pmi in pmi_very],
        'pmi_edm': [pmi for pmi in pmi_edm ], 
    })
    frame = frame.replace([np.inf, -np.inf], 0) # TODO: should not need this anymore since there is add-1 smoothing.
    frame = frame.drop_duplicates(subset=['word'], keep='first')
    frame.set_index('word', inplace=True)
    return frame