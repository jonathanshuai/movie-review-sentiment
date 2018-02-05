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
from sklearn.decomposition import PCA

from gensim.models import Word2Vec

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


VOCAB_SIZE = 2400
N_EPOCHS = 100
MAX_NORM = 1e1

print("Processing text...")

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

print("Finished processing text! Creating word vectors...")

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

X = []
for review in reviews:
  X.append(np.array([model.get_vector(word) for word in review if word in model.vocab]))

# Split into training and testing set
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Finished creating word vectors! Training RNN...")

# Simple RNN
class RNN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=128, p=0.08):
    super(RNN, self).__init__()
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    self.dropout = nn.Dropout(p=p)
    self.leaky_relu = nn.LeakyReLU(0.01)
    
    self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)

  def forward(self, word, hidden):
    combined = torch.cat((word, hidden), 1)
    combined = self.leaky_relu(combined)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    log_probs = F.log_softmax(output, dim=1)
    return log_probs, hidden

# Train the model
loss_function = nn.NLLLoss()
model = RNN(100, 2)
optimizer = optim.SGD(model.parameters(), lr=0.0008, weight_decay=1e-3)

n_train_samples = len(X_train)
# For n epochs...
for epoch in range(N_EPOCHS):
  total_loss = torch.Tensor([0])

  random_indices = np.random.permutation(n_train_samples)

  for index in random_indices:
    review = X_train[index]
    label = int(y_train[index]) # Why doesn't y.astype(int) work??

    # Initialize hidden layer
    hidden = autograd.Variable(torch.zeros((1, 128)))

    # Get the word vectors and feed them into the RNN w/ forward propagation
    word_vector = autograd.Variable(torch.from_numpy(review))
    model.zero_grad()
    for w in range(word_vector.size()[0]):
      word = word_vector[w].view(1, -1)
      output, hidden = model(word, hidden)
    
    # Calculate Loss
    loss = loss_function(output, autograd.Variable(torch.LongTensor([label])))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), MAX_NORM)
    optimizer.step()
    total_loss += loss.data

  print(torch.norm(next(model.parameters()).grad))
  print("[epoch {}] {}".format(epoch, total_loss))

preds = []
for review, label in zip(X_train, y_train):
  hidden = autograd.Variable(torch.zeros((1, 128)))
  word_vector = autograd.Variable(torch.from_numpy(review))
  model.zero_grad()
  for w in range(word_vector.size()[0]):
    word = word_vector[w].view(1, -1)
    output, hidden = model(word, hidden)
  preds.append(np.argmax(output.data.numpy()))

train_accuracy = accuracy_score(preds, y_train)

preds = []
for review, label in zip(X_test, y_test):
  hidden = autograd.Variable(torch.zeros((1, 128)))
  word_vector = autograd.Variable(torch.from_numpy(review))
  model.zero_grad()
  for w in range(word_vector.size()[0]):
    word = word_vector[w].view(1, -1)
    output, hidden = model(word, hidden)
  preds.append(np.argmax(output.data.numpy()))

test_accuracy = accuracy_score(preds, y_test)




