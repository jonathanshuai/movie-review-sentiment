##Predicting Movie Review Sentiment
Data is from [Cornell](http://www.cs.cornell.edu/people/pabo/movie-review-data)

Simple bash command to create a csv to find the paths to the text and make life easier:
```
echo review_path,label > data.csv; find -name *.txt | awk -v OFS=',' '{if ($1 ~ /pos/) {print $1, 0} else {print $1, 1}}' >> data.csv
```

###Bag of Words Model
Used a bag of words model with Gaussian Naive Bayes and SVM. Example of results from cross validation:
```
-----VOCABULARY SIZE 3600-----
[Gaussian Bayes] Training accuracy: 0.87, Testing accuracy: 0.76
[SVC] Training accuracy: 0.97, Testing accuracy: 0.85
kernel: rbf, C: 10
```
This is not bad, and can definitely be improved with more tweaking. But I want to see if deep learning will do even better.

###RNN
Trained this model for 30 epochs. Learning error (cross entropy loss) was around ~140 for 1500 samples was still going down. Expected loss for random guessing is 1500 * -log(0.5) ~= 1040.

Accuracy on training set >95%
Accuracy on testing set <55%
(serious overfitting issues)

Try something like dropout:
