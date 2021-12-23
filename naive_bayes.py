#!/usr/bin/python3

"""
Objective: Train and test bigram and tigram models

Reference: Chapter 3, N-Gram Language Models,
Speech and Language Processing, Dan Jurasky and James Martin, 2020
"""

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords

def load_words():
    df = pd.read_csv('data/words.csv')
    stop = stopwords.words('english')
    df = df[~df['words'].isin(stop)].reset_index(drop=True)
    return df

def get_N_grams(words, n=2):
    n_grams = []
    for word in words:
        word = ' '*(n-1) + word + ' '*(n-1)
        for i in range(0, len(word)-(n-1)):
            n_grams.append(word[i:i+n])
    return Counter(n_grams)

def get_n_gram_probability(n_gram):
    return n_grams[n_gram] / N

def evaluate(words):
     
    return

if __name__ == "__main__":
    df = load_words()
    train_words = df.sample(int(0.8 * df.shape[0]))
    test_words = df.drop(train_words.index)
    n_grams = get_N_grams(train_words['words'])
    N = sum(n_grams.values())
    evaluate(test_words['words'])
