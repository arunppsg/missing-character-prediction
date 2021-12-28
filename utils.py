#!/usr/bin/python

"""
Objective: Predict missing characters in a word
"""

import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.model_selection import KFold

def load_words():
    df = pd.read_csv('data/words.csv')
    stop = stopwords.words('english')
    df = df[~df['words'].isin(stop)].reset_index(drop=True)
    return df

def find_missing_character(word):
    candidate_words = get_candidate_words(word)
    if len(candidate_words) == 0:
        return None
    probabilities = list(map(word_probability, candidate_words))
    return candidate_words[probabilities.index(min(probabilities))]

def remove_random_character(word):
    """
    Chooses an incorrect word position and fills it with a random alphabet
    """
    pos = np.random.randint(len(word))
    incorrect_token = '-'
    incorrect_word = word[:pos] + incorrect_token + word[pos+1:]
    return incorrect_word

def log_incorrect_pairs(incorrect_pairs):
    with open('incorrect_pairs.txt', 'a') as f:
        for word, incorrect_word in incorrect_pairs:
            f.write("{} - {}\n".format(word, incorrect_word))

if __name__ == "__main__":
    df = load_words()
    kf = KFold(n_splits=10, random_state=200, shuffle=True)
    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_words = df.loc[train_idx]['words']
        test_words = df.loc[test_idx]['words']
        WORDS = Counter(train_words)
        N = sum(WORDS.values())
        score = evaluate(test_words, 1)
        scores.append(score)
    print ("Average score is %f " % np.mean(scores)) 
