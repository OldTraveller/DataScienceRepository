print("LABELLING OF DATA IN PYTHON") 

# Labelling of the data can be done using the Encoder in preprocessing module. 
# Labelling can be used to transform a given data in one form to the other form. 
# In sklearn preprocessing requires the data to be labeeled in the number system. 
# Labelling has to be done before sending the data to any Machine Learning Algorithm. 
# Transforming the data from the label form to the numerical form is called LABEL
# ENCODING. 
# ----------------------------------------------------------------------------------
#                       Steps in Label Encoding.
# ----------------------------------------------------------------------------------
# import the proper libraries
import numpy as np 
import sklearn.preprocessing as preprocessing
 
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white'] 

# Creating an training of label encoder. 
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels) 
encoded_values = encoder.transform(input_labels)
print("Labels : ", input_labels)
print("Encoded Values : ", encoded_values) 

decode_what = [0, 1, 2, 3] 
decoded_labels = encoder.inverse_transform(decode_what) 
print("Decoded Numbers : ", decode_what)
print("Decode Labels : ", decoded_labels) 