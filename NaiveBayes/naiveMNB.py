#Import Library
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Create SVM classification object
clf = MultinomialNB()
clf.fit(X,Y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

y_pred = clf.predict(iris.data)
print(y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))


