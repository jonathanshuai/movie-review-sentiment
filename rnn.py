import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


VOCAB_SIZE = 2400
N_EPOCHS = 50
MAX_NORM = 1e4

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
# Split into training and testing set
y = df['label']

reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, y)

word_count = Counter('')
for review_text in reviews_train:
  word_count += Counter(review_text)

# Get a dictiona ry of our vocabulary w/ indices
vocabulary = {}
for i, (w, c) in enumerate(word_count.most_common(VOCAB_SIZE - 1)):
  vocabulary[w] = i

# Turn reviews into 1-hot encoded vectors
X_train = []
for review in reviews_train:
  X_train.append([vocabulary[w] if w in vocabulary else VOCAB_SIZE - 1 for w in review])

X_test = []
for review in reviews_test:
  X_test.append([vocabulary[w] if w in vocabulary else VOCAB_SIZE - 1 for w in review])
  
print("Finished creating word vectors! Training RNN...")

# Simple RNN
class RNN(nn.Module):
  def __init__(self, vocab_size, output_dim, embed_dim=256, hidden_dim=128):
    super(RNN, self).__init__()
    self.vocab_size = vocab_size
    self.output_dim = output_dim
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim

    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    self.i2h = nn.Linear(embed_dim + hidden_dim, hidden_dim)
    self.i2o = nn.Linear(embed_dim + hidden_dim, output_dim)

  def forward(self, word, hidden):
    embeds = self.embeddings(word)    
    combined = torch.cat((embeds, hidden), 1)
    hidden = self.i2h(combined)
    output = F.log_softmax(self.i2o(combined), dim=1)
    return output, hidden

# Train the model
loss_function = nn.NLLLoss()
model = RNN(VOCAB_SIZE, 2)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# For n epochs...
for epoch in range(N_EPOCHS):
  total_loss = torch.Tensor([0])
  for review, label in zip(X_train, y_train):
    # Initialize hidden layer
    hidden = autograd.Variable(torch.zeros((1, 128)))
    word_vector = autograd.Variable(torch.LongTensor(review))
    model.zero_grad()
    for w in range(word_vector.size()[0]):
      output, hidden = model(word_vector[w], hidden)
    
    loss = loss_function(output, autograd.Variable(torch.LongTensor([label])))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), MAX_NORM)
    optimizer.step()
    total_loss += loss.data
  print(torch.norm(next(model.parameters()).grad))
  print("[epoch {}] {}".format(epoch, total_loss))

#print(losses)  # The loss decreased every iteration over the training data!



preds = []
for review, label in zip(X_train, y_train):
  hidden = autograd.Variable(torch.zeros((1, 128)))
  word_vector = autograd.Variable(torch.LongTensor(review))
  model.zero_grad()
  for w in range(word_vector.size()[0]):
    output, hidden = model(word_vector[w], hidden)
  preds.append(output)

train_accuracy = accuracy_score(preds, y_train)

preds = []
for review, label in zip(X_test, y_test):
  hidden = autograd.Variable(torch.zeros((1, 128)))
  word_vector = autograd.Variable(torch.LongTensor(review))
  model.zero_grad()
  for w in range(word_vector.size()[0]):
    output, hidden = model(word_vector[w], hidden)
  preds.append(output)

test_accuracy = accuracy_score(preds, y_test)

