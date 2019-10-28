import numpy as np 
import sklearn
# Loading the already present dataset into python3 program
from sklearn.datasets import load_breast_cancer 
# Used to split the entire data into training data and the testing data 
from sklearn.model_selection import train_test_split

# Classification Problem basically deals with categorising a given data
# in the form of the classes that is provided. 
# It can be a black and white class 
# Teaching or Non Teaching class 
# Big Spender or Budget Spender or any such similar class
# A number of dataset is been targetted and fed inside and then found the output off. 
# Then using these the unclassified data is fed and results been classified. 

data = load_breast_cancer() 
label_names = data['target_names'] 
labels = data['target'] 
feature_names = data['feature_names'] 
features = data['data'] 

# To display the feature names. 
# for i in feature_names: 
#         print(i)

train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.50, random_state = 42)

# Now we will be building our model using Naive Bayesian Algorithm to build the model. 
# We can import the already present algorithm from the Scikit Learn Library. 
from sklearn.naive_bayes import GaussianNB 

# Initialize the model
gnb = GaussianNB() 
model = gnb.fit(train, train_labels) 

# Now predicting on the test data. 
predictions = gnb.predict(test)
print("Printing the prediction Results:")
print(predictions) 

# Now the data has been predicted. 
# From the already exisiting results in the test_labels and the results
# that is predicted now. We will be testing the accuracy of the stuff.
# This can be done using one of the modules which is present which is called - 
# accuracy_score() 
from sklearn.metrics import accuracy_score 
print("Accuracy Attained is : ", accuracy_score(test_labels, predictions)) 

