#!/usr/bin/python3

"""
Objective: Train and test bigram and tigram models

Reference: Chapter 3, N-Gram Language Models,
Speech and Language Processing, Dan Jurasky and James Martin, 2020
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import KFold

import utils

def get_N_grams(words, n=2):
    n_grams = []
    for word in words:
        word = ' '*(n-1) + word + ' '*(n-1)
        for i in range(0, len(word)-(n-1)):
            n_grams.append(word[i:i+n])
    n_gram_counts = Counter(n_grams)

    prev_chars = []  # w_{1:n-1}
    last_char = []  # w_n
    count = []

    for n_gram in n_gram_counts:
        prev_chars.append(n_gram[:n-1])
        last_char.append(n_gram[n-1])
        count.append(n_gram_counts[n_gram])

    df = pd.DataFrame({'context':prev_chars, 'target':last_char, 'count':count})
    df['prob'] = df['count'] / df['count'].sum()

    ngram_model = defaultdict(lambda: defaultdict(lambda: float()))
    for idx, row in df.iterrows():
        ngram_model[row['context']][row['target']] = row['prob']

    return ngram_model

def evaluate(test_words, ngram_model, n=2):
    possibilities = '-abcdefghijklmnopqrstuvwxyz'
    score = 0
    
    for word in test_words:
        incorrect_word = utils.remove_random_character(word)
        incorrect_word = ' '*(n-1) + incorrect_word + ' '*(n-1)
        
        pos = incorrect_word.find('-')
        context = incorrect_word[pos-n+1:pos]
        
        probs = {}
        for char in possibilities:
            probs[char] = ngram_model[context][char]
            
        if len(probs) > 0:
            missing_char = max(probs, key=probs.get)
            correction = incorrect_word.replace('-', missing_char).strip(' ')
            if correction == word:
                score += 1
                
    return score / len(test_words)

if __name__ == "__main__":
    df = utils.load_words()
    kf = KFold(n_splits=10, random_state=200, shuffle=True)
    scores = defaultdict(list) 
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_words = df.loc[train_idx]['words']
        test_words = df.loc[test_idx]['words']

        bigram_model = get_N_grams(train_words, n=2)
        trigram_model = get_N_grams(train_words, n=3)
        quadgram_model = get_N_grams(train_words, n=4)
        pentgram_model = get_N_grams(train_words, n=5)
        hexgram_model = get_N_grams(train_words, n=6)

        scores['bigram_model'].append(evaluate(test_words, bigram_model, 2))
        scores['trigram_model'].append(evaluate(test_words, trigram_model, 3))
        scores['quadgram_model'].append(evaluate(test_words, quadgram_model, 4))
        scores['pentgram_model'].append(evaluate(test_words, pentgram_model, 5))
        scores['hexgram_model'].append(evaluate(test_words, hexgram_model, 6))
 

    with open('results/n_gram.txt', 'a') as f:
        f.write("Results of 10-fold cross validation")
        f.write("\nAverage score of bigram model is {:.2f}".format(np.mean(scores['bigram_model'])))
        f.write("\nAverage score of trigram model is {:.2f}".format(np.mean(scores['trigram_model'])))
        f.write("\nAverage score of quadgram model is {:.2f}".format(np.mean(scores['quadgram_model'])))
        f.write("\nAverage score of pentgram model is {:.2f}".format(np.mean(scores['pentgram_model'])))
        f.write("\nAverage score of hextgram model is {:.2f}".format(np.mean(scores['hexgram_model'])))
