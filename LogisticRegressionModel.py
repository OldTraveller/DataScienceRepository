# Install Tkinter Package 
import numpy as np 
from sklearn import linear_model 
import matplotlib.pyplot as plt 

X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4], [3.2, 0.9], [2.9, 1.9],[2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C = 75)
Classifier_LR.fit(X, y) 

def LogisticVisualize(Classifier_LR, X, y): 
        min_x = X[:, 0].min() - 1.0
        max_x = X[:, 0].max() + 1.0 
        min_y = X[:, 1].min() - 1.0 
        max_y = X[:, 1].max() + 1.0
        mesh_step_size = 0.02
        