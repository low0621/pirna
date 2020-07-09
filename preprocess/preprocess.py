from itertools import product
import pandas as df
import numpy as np

def ngram_encode(seq):
    N_GRAM = 1
    def index2list(i):
        r = [0] * (4 ** N_GRAM)
        r[i] = 1
        return r
    lut = { e: index2list(i) for (i, e) in enumerate(product(['A', 'T', 'C', 'G'], repeat=N_GRAM))}
    result = []
    for i in range(len(seq) // N_GRAM):
        key = tuple(seq[i * N_GRAM: (i+1) * N_GRAM])
        result.append(lut[key])
    return np.array(result)

def df2arr(train_data=None, valid_data=None, test_data=None, mode='train'):
    if 'train' == mode:
        pi_train = train_data['piRNA_seq'].map(lambda x: ngram_encode(x)).to_numpy()
        m_train = train_data['mRNA_site'].map(lambda x: ngram_encode(x)).to_numpy()
        y_train = train_data['label'].to_numpy()

        pi_val = valid_data['piRNA_seq'].map(lambda x: ngram_encode(x)).to_numpy()
        m_val = valid_data['mRNA_site'].map(lambda x: ngram_encode(x)).to_numpy()
        y_val = valid_data['label'].to_numpy()

    pi_test = test_data['piRNA_seq'].map(lambda x: ngram_encode(x)).to_numpy()
    m_test = test_data['mRNA_site'].map(lambda x: ngram_encode(x)).to_numpy()

    if 'test' == mode:
        return pi_test, m_test

    return pi_train, pi_val, pi_test, m_train, m_val, m_test, y_train, y_val
