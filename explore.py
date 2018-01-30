import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

VOCAB_SIZE = 2000

data_file = 'data.csv'


df = pd.read_csv(data_file)

#read and preprocess the review text
stop_words = set(stopwords.words('english'))
punctuation = '\'"-:@#$%^(){}[]_=+<>|/\\~' #get rid of punctuation except . , ! ;

word_count = Counter('')
reviews = []
for review_path in df['review_path']:
  # Open the file
  with open(review_path, 'r') as f:
    review_text = f.read()
    review_text = review_text.lower()

    #Filter out punctuation
    filter_punct = []
    for c in review_text:
      if c not in punctuation:
        filter_punct.append(c)

    review_text = ''.join(filter_punct)
    review_text = word_tokenize(review_text)

    # Remove stop words 
    filter_stop_words = []
    for w in review_text:
      if w not in stop_words:
        filter_stop_words.append(w)
    review_text = filter_stop_words

    # Append the wordcount and the filtered review text    
    word_count += Counter(review_text)
    reviews.append(review_text)

# Get a ditionary of our vocabulary
vocabulary = dict(word_count.most_common(VOCAB_SIZE))



























