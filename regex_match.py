#!/usr/bin/python3

"""
Naive Pattern matching using Regex 

"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold

import utils

def evaluate(test_words, corpus):
    score = 0
    for word in test_words:
        incorrect_word = utils.remove_random_character(word)
        pat = incorrect_word.replace('-', '.')
        matches = re.findall(pat, corpus)

        if len(matches) > 0:
            correction = Counter(matches).most_common(1)[0][0]
            if correction == word:
                score += 1

    return score / len(test_words)

if __name__ == "__main__":
    df = utils.load_words()
    kf = KFold(n_splits=10, random_state=200, shuffle=True)
    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_words = df.loc[train_idx]['words']
        test_words = df.loc[test_idx]['words']

        train_corpus = ' '.join(train_words.unique())
        scores.append(evaluate(test_words, train_corpus))  
    print ("Average Score {:.2f}".format(np.mean(scores)))
    with open('results/regex.txt', 'a') as f:
        f.write("Results of 10-fold cross validation")
        f.write("\nAverage score of is {:.2f}".format(np.mean(scores))) 
