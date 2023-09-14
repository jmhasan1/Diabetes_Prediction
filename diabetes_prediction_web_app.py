# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 22:29:46 2023

@author: Jahid Hasan
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('D:\STUDY\DATA SCIENCE , ML , ANALYSIS\ML Projects\Diabetese Prediction\trained_model.sav', 'rb'))


# creating a function for prediction

def diabetes_prediction(input_data):
    
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
        return "The person is not diabetic"
    else:
        return "The patient is diabetic"
    
    
    
def main():
    
    
    #giving a title
    st.title("Diabetes Prediction Web App:")
    
    
    #getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies: ')
    Glucose = st.text_input('Glucose Level: ')
    BloodPressure = st.text_input('Blood Pressure : ')
    SkinThickness = st.text_input('SkinThickness Value: ')
    Insulin = st.text_input('Insulin Unit: ')
    BMI = st.text_input('BMI: ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value: ')
    Age = st.text_input('Age: ')
    
    
    #code fpr prediction
    diagnosis = ''
    
    
    #creating a button for prediction 
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
        
    
    
if __name__  == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    