# Import Library
import numpy as np
import matplotlib.pyplot as pt
from sklearn import linear_model
from sklearn import datasets, linear_model

# Load Train and Test datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train = diabetes_X[:-20] #input_variables_values_training_datasets
x_test = diabetes_X[-20:] #target_variables_values_training_datasets

y_train = diabetes.target[:-20] #target_variables_values_training_datasets
y_test= diabetes.target[-20: ]#input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)

pt.scatter(x_test, y_test, color='green')
pt.plot(x_test, predicted, color='blue')

pt.xticks(())
pt.yticks(())

pt.show()
