import streamlit as st
import pandas as pd
import joblib

#Load the model and scaler from the modelling notebook
model = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Olist Bad Review Predictor')
st.write('Enter the details of a delivered order to check the risk of a bad review (1-2 stars).')

#Inputs from the user
estimated_days = st.number_input('Promised delivery time (days)', min_value=1, value=25)
delivery_days = st.number_input('Actual delivery time (days)', min_value=0, value=33)
n_items = st.number_input('Number of items', min_value=1, value=1)
n_sellers = st.number_input('Number of sellers', min_value=1, value=1)
total_price = st.number_input('Total price (BRL)', min_value=0.0, value=100.0)
total_freight = st.number_input('Total freight (BRL)', min_value=0.0, value=20.0)

#Calculate the same features as in the EDA notebook
delay_days = delivery_days - estimated_days
is_late = 1 if delay_days > 0 else 0

if st.button('Predict'):
    #Same column names and order as the features used for training
    order = pd.DataFrame([{
        'delivery_days': delivery_days, 'estimated_days': estimated_days,
        'delay_days': delay_days, 'is_late': is_late,
        'n_items': n_items, 'n_sellers': n_sellers,
        'total_price': total_price, 'total_freight': total_freight}])

    risk = model.predict_proba(scaler.transform(order))[:, 1][0]

    st.metric('Risk score', f'{risk:.1%}')
    if risk >= 0.5:
        st.error('High risk: customer service should contact this customer.')
    else:
        st.success('Low risk: no action needed.')