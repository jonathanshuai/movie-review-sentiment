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


VOCAB_SIZES = [600, 1200, 2400, 3600, 4800]
N_EPOCHS = 100

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





# Simple RNN
class RNN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=128):
    super(RNN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)

  def forward(self, word_vector, hidden):
    combined = torch.cat((word_vector, hidden), 1)
    hidden = self.i2h(combined)
    output = F.log_softmax(self.i2o(combined), dim=1)
    return output, hidden


self.hidden = autograd.Variable(torch.randn((1, hidden_dim)))

losses = []
loss_function = nn.NLLLoss()
model = RNN(VOCAB_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(N_EPOCHS):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()

        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!








