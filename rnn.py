import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

VOCAB_SIZES = [600, 1200, 2400, 3600, 4800]

# Read in the labels and paths for reviews
data_file = 'data.csv'
df = pd.read_csv(data_file)

stop_words = set(stopwords.words('english'))
punctuation = '\'"-:@#$%^(){}[]_=+<>|/\\~' #get rid of punctuation except . , ! ;

# Read in the review text and preprocess it
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

    # Append filtered review text    
    reviews.append(review_text)

# Split into training and testing set
y = df['label']
reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, y)

word_count = Counter('')
for review_text in reviews_train:
  word_count += Counter(review_text)

# Get a dictionary of our vocabulary w/ indices
vocabulary = {}
for i, (w, c) in enumerate(word_count.most_common()):
  vocabulary[w] = i

#Test w/ different vocabulary sizes
for VOCAB_SIZE in VOCAB_SIZES: