import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.plk', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder_geo.plk','rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open('scaler.plk', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18, 92)

balance = st.number_input('Balance', min_value=0, step=1, format="%d")
credit_score = st.number_input('Credit Score', min_value=0, step=1, format="%d")
estimated_salary = st.number_input('Estimated Salary', min_value=0, step=1, format="%d")


tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Product', 1, 4)
has_cr_card = int(st.checkbox('Has Credit Card'))
is_active_member = int(st.checkbox('Is Active Member'))

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance' : [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())
# input_data = pd.DataFrame([[input_data]])

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')