#Import Library
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

iris = datasets.load_iris()

# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
model = GaussianNB()
# Train the model using the training sets and check score
y_pred = model.fit(iris.data, iris.target).predict(iris.data)

print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
