import streamlit as st
import pandas as pd
import pickle
import joblib
#from PIL import Image

model = joblib.load("classification2.pkl")

st.header("# Kowope Mart Predict Defaulters App")

product_type = st.sidebar.slider('product_type', 1.0, 2.0)
credit_risk = st.sidebar.slider('credit_risk', 1.0, 3.0)
customer_Creditworthiness = st.sidebar.slider('customer_Creditworthiness', 1.0, 6.9, 1.3)
    
data = {'product_type': product_type,
            'credit_risk': credit_risk,
            'customer_Creditworthiness': customer_Creditworthiness
            }
features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Probablity of default status being No ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of default status being Yes is ( in % )**:',pred_proba[0][1]*100)