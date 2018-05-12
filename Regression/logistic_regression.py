# Import Library
import numpy as np
import matplotlib.pyplot as pt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, :2] # 2 features
Y = iris.target

h = .015 # step size in the mesh

# Create logistic regression object
model = LogisticRegression(C=1e5)

# Train the model using the training sets and check score
model.fit(X, Y)
model.score(X, Y)

#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)


# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Predict Output
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pt.figure(1, figsize=(4, 3))
pt.pcolormesh(xx, yy, Z, cmap=pt.cm.Paired)

# Plot also the training points
pt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pt.cm.Paired)
pt.xlabel('Sepal length')
pt.ylabel('Sepal width')

pt.xlim(xx.min(), xx.max())
pt.ylim(yy.min(), yy.max())
pt.xticks(())
pt.yticks(())

pt.show()

