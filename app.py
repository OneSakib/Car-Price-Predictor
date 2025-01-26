import locale
import streamlit as st
import pickle
import pandas as pd
import sklearn

df = pd.read_csv('cleaned car.csv')
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
st.title("Car Price Prediction")
company_name = st.selectbox(
    "Select the company?",
    sorted(df['company'].unique())
)
model_name = st.selectbox(
    "Select the model?",
    sorted(df['name'].unique())
)
year_name = st.selectbox(
    "Select the Year?",
    sorted(df['year'].unique(), reverse=True)
)
fuel_type_name = st.selectbox(
    "Select the Fuel Type?",
    sorted(df['fuel_type'].unique())
)
no_of_kms = st.number_input("Enter the number of kms")
locale.setlocale(locale.LC_ALL, 'en_IN')
if st.button("Predict"):
    predict_price = model.predict(pd.DataFrame([[model_name, company_name, year_name, no_of_kms, fuel_type_name]], columns=[
                                  'name', 'company', 'year', 'kms_driven', 'fuel_type']))
    inr_format = locale.currency(predict_price[0], grouping=True, symbol="₹")
    st.write(f"The predicted price is {inr_format}")
