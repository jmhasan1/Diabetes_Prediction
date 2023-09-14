# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 
# loading the saved model
loaded_model = pickle.load(open('D:/STUDY/DATA SCIENCE , ML , ANALYSIS/ML Projects/Diabetese Prediction/trained_model.sav', 'rb'))


input_data = (1,89,66,23,94,28.1,0.167,21)

# changing the input-data to numpy array
np_input_data = np.asarray(input_data)

# reshaping the array as we predicting for only one instance
rs_np_input_data = np_input_data.reshape(1,-1)

# standardize the input data
# std_data = scaler.transform(rs_np_input_data)
# print("Standarized input data: ",std_data)

prediction = loaded_model.predict(rs_np_input_data)
print(prediction)

if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The patient is diabetic")