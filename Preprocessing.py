import numpy as np
import sklearn.preprocessing as preprocessor 

# After importing the suitable packages 
input_data = np.array([ [2.1, -1.9, 5.5], 
                        [-1.5, 2.4, 3.5],
                        [0.5, -7.9, 5.6], 
                        [5.9, 2.3, -5.8] ])

# ----------------------------------------------------------------------------------
#                                Binarization
# ----------------------------------------------------------------------------------
# This is the preprocessing technique which is used when 
# we need to convert our numerical values into Boolean values. We can use an inbuilt method 
# to binarize the input data say by using 0.5 as the threshold value in the following way −
binarized_data = preprocessor.Binarizer(threshold=0.5).transform(input_data)
print("Binarized Data is ")
print(binarized_data) 

# ----------------------------------------------------------------------------------
#                                Mean Removal
# ----------------------------------------------------------------------------------
# It is another very common preprocessing technique, that is used in Machine Learning. 
# To get the mean of the vector 
print("Mean = ", input_data.mean(axis = 0)) 
print("Standard Deviation = ", input_data.std(axis = 0)) 
# To get the mean of the binarized data we send the Binarized Data as input
print("Mean : ", binarized_data.mean(axis = 0)) 
print("Standard Deviation : ", binarized_data.std(axis = 0)) 
# So for each dimension the set of values are created. 
# That is for each of the columns. We get the values. 

# ----------------------------------------------------------------------------------
#                        SCALING - Preprocessing Technique 
# ----------------------------------------------------------------------------------
# Since we do not want some of the values of the feature vectors to exceed beyond 
# something which might not be desired. For that we might want to restrict them to 
# a range of equivalend values which can be used for computation. 
# Eg -  The values of marks of 700 out of 1000 and 650 out of 1000 can be reduced within
#       the range of [0, 1] by dividing by 1000. So we have the new values of the values 
#       as 0.700 and 0.650 respectively which is just a small in the range value. 
data_scaler_minmax = preprocessor.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) 
print("MinMax Scaled version of the data in the range is : ") 
print(data_scaled_minmax) 

# ----------------------------------------------------------------------------------
#                       NORMALIZATION 
# ----------------------------------------------------------------------------------
# It is also referred to as Least Absolute Deviations. The kind of normalization modifies the
# values so that the sum of absoulte values is always = 1 in each row. 
data_normalized_l1 = preprocessor.normalize(input_data, norm='l1') 
print("N1 Normalized Data : ")
print(data_normalized_l1) 

# Least Squares - such that the sum of the squares is = 1 in each row. 
data_normalized_l2 = preprocessor.normalize(input_data, norm='l2') 