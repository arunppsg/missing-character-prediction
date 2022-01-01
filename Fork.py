import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import io
from nltk.corpus import stopwords
from sklearn.model_selection import KFold

def remove_random_character(word):
    """
    Chooses an incorrect word position and fills it with a random alphabet
    """
    pos = np.random.randint(len(word))
    incorrect_token = '-'
    incorrect_word = word[:pos] + incorrect_token + word[pos+1:]
    return incorrect_word

def load_words():
    df = pd.read_csv(io.BytesIO(uploaded['words.csv']))
    stop = stopwords.words('english')
    df = df[~df['words'].isin(stop)].reset_index(drop=True)
    return df

def get_N_grams(words, n):
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

def evaluate(test_words, ngram_model, n):
    possibilities = '-abcdefghijklmnopqrstuvwxyz'
    score = 0
    
    for word in test_words:
        incorrect_word = remove_random_character(word)
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

def predicited_value(context, ngram_model):
  possibilities = 'abcdefghijklmnopqrstuvwxyz'
  probs = {}
  for char in possibilities:
    probs[char] = ngram_model[context][char]
  if len(probs) > 0:
    missing_char = max(probs, key=probs.get)
    output= context + missing_char 
  return output

if __name__ == "__main__":
    df = load_words()
    kf = KFold(n_splits=10, random_state=200, shuffle=True)
    scores = defaultdict(list) ; score={}
    for i, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_words = df.loc[train_idx]['words']
        test_words = df.loc[test_idx]['words']
        model={}
        for j in range(2,10):
          model[j] = get_N_grams(train_words, n=j)
        #bigram_model = get_N_grams(train_words, n=2)
        #trigram_model = get_N_grams(train_words, n=3)
        #quadgram_model = get_N_grams(train_words, n=4)
        #pentgram_model = get_N_grams(train_words, n=5)
        #hexgram_model = get_N_grams(train_words, n=6)

        for j in range(2,10):
          scores[j].append(evaluate(test_words, model[j], j))
        #scores['1'].append(evaluate(test_words, bigram_model, 2))
        #scores['2'].append(evaluate(test_words, trigram_model, 3))
        #scores['3'].append(evaluate(test_words, quadgram_model, 4))
        #scores['4'].append(evaluate(test_words, pentgram_model, 5))
        #scores['5'].append(evaluate(test_words, hexgram_model, 6))

key=list(scores.keys())
for i in key:
  score[i]=np.mean(scores[i])
sorted_score = sorted(score, key=score.get, reverse=True)  
model=get_N_grams(df['words'], sorted_score[0])
predict=predicited_value('lowin',model) 

   
