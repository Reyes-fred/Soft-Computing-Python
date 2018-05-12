#Import Library
import numpy as np
import matplotlib.pyplot as pt
from sklearn import svm,datasets

def meshgrid(x, y , h=.02):
   """Parameters
      X : Data to base x-axis
      Y : Data to base y-axis
      h : Stepsize for meshgrid optional
   """
   x_min, x_max = x.min() -1, x.max() + 1
   y_min, y_max = y.min() -1, y.max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))
   return xx, yy

def contours(ax, clf, xx, yy, **params):
   """ Parameters
      ax : Matplotlib axes object
      clf : a classifier
      xx : meshgrid ndarray
      yy : meshgrid ndarray
      params: dictionary of params to pass to contourf - optional
   """

    
#Import data from iris database
iris = datasets.load_iris()
X = iris.data[:, :2] #Two features
Y = iris.target

C = 1.0 #SVM regulariztion parameters

# Create SVM classification object 
models = (svm.SVC(kernel='linear', C=C),svm.LinearSVC(C=C),svm.SVC(kernel='rbf', gamma=0.7, C=C),svm.SVC(kernel='poly',degree=3, C=C))
model = (clf.fit(X, Y) for clf in models)

titles = ('SVC with linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 3) kernel')

fig,sub = pt.subplots(2,2)
pt.subplots_adjust(wspace=0.4,hspace=0.4)

X0, X1 = X[:,0], X[:,1]
xx, yy = meshgrid(X0,X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    contours(ax, clf, xx, yy,
                  cmap=pt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=pt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

pt.show()
