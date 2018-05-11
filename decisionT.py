#Import Library
import numpy as np
from sklearn import model_selection
from sklearn import datasets
from sklearn import tree

iris = datasets.load_iris()
X = iris.data
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=0)

# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  

# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, Y)
model.score(X, Y)

#Predict Output
predicted= model.predict(x_test)
