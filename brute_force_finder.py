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

def word_probability(word):
    """
    Returns probability of `word`
    """
    return WORDS[word]/N

def get_candidate_words(word):
    """
    Returns most probable candidate word
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    incorrect_token = '-'
    candidate_words = []
    for letter in letters:
        candidate_word = word.replace(incorrect_token, letter)
        if candidate_word in WORDS:
           candidate_words.append(candidate_word) 
    return candidate_words

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

def evaluate(words, log=0):
    score = 0
    incorrect_pairs = []
    for word in words:
        incorrect_word = remove_random_character(word)
        possible_word = find_missing_character(incorrect_word)
        if word == possible_word:
            score += 1
        else:
            incorrect_pairs.append((word, possible_word))
    if log == 1:
        log_incorrect_pairs(incorrect_pairs)
    total_score = score / len(words)
    return total_score 

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
