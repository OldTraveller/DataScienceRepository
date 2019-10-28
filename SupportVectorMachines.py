# Support vector machine is a classification as well as regression
# Machine Learning Algorithm. The concept of SVM is to plot each feature 
# as a point in n dimensional space. 
# There is a line which separates the points into two classified groups. 
# This is known as the classifier. 

# ----------------------------------------------------------------------------------
# Here we are going to build an iris classifier using SVM
# ----------------------------------------------------------------------------------
# Scikit Learn library has a svm module. 
# sklearn.svm.svc for classification.

# ----------------------------------------------------------------------------------
# The Iris Dataset provided in the sklearn library
# ----------------------------------------------------------------------------------
# It has 3 classes of 50 instances each. 

import pandas as pd 
import numpy as np 
from sklearn import svm, datasets
import matplotlib.pyplot as plt 

# Load the dataset 
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

svm_classifier = svm.LinearSVC()
Svc_classifier = svm_classifier.SVC(kernel='linear', 
C= 1.0 , decision_function_shape = 'ovr').fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize = (15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')