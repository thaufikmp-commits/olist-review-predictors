import streamlit as st
import pandas as pd
import joblib

# Page settings: browser tab title, icon and wide layout
st.set_page_config(page_title='Olist Review Predictor', layout='wide')

# Load the model and scaler from the modelling notebook
model = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------- Sidebar: information about the model ----------
st.sidebar.header('About the model')
st.sidebar.write('Logistic Regression trained on 76,659 delivered Olist orders.')
st.sidebar.write('**Test results**')
st.sidebar.write('- Recall: 0.51\n- Precision: 0.39\n- F1: 0.44\n- ROC-AUC: 0.75')
st.sidebar.caption('Data: Brazilian E-Commerce Public Dataset by Olist (Kaggle)')

# ---------- Header ----------
st.title('Olist Bad Review Predictor')
st.write('Enter the details of a delivered order. The model estimates the risk that '
         'the customer leaves a bad review (1–2 stars), so customer service can act early.')

# ---------- Inputs in two columns ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader('Delivery')
    estimated_days = st.number_input('Promised delivery time (days)', min_value=1, value=25)
    delivery_days = st.number_input('Actual delivery time (days)', min_value=0, value=33)

with col2:
    st.subheader('Order')
    n_items = st.number_input('Number of items', min_value=1, value=1)
    n_sellers = st.number_input('Number of sellers', min_value=1, value=1)
    total_price = st.number_input('Total price (BRL)', min_value=0.0, value=100.0)
    total_freight = st.number_input('Total freight (BRL)', min_value=0.0, value=20.0)

# Calculate the same features as in the EDA notebook
delay_days = delivery_days - estimated_days
is_late = 1 if delay_days > 0 else 0

# Show the delivery status straight away
if is_late:
    st.warning(f'This order arrived **{delay_days} days late**.')
else:
    st.info(f'✅ This order arrived **{-delay_days} days early** (or on time).')

# ---------- Prediction ----------
if st.button('Predict risk', type='primary'):
    order = pd.DataFrame([{
        'delivery_days': delivery_days, 'estimated_days': estimated_days,
        'delay_days': delay_days, 'is_late': is_late,
        'n_items': n_items, 'n_sellers': n_sellers,
        'total_price': total_price, 'total_freight': total_freight}])

    risk = model.predict_proba(scaler.transform(order))[:, 1][0]

    st.subheader('Result')
    st.metric('Risk score', f'{risk:.1%}')
    st.progress(float(risk))   # visual bar from 0% to 100%

    if risk >= 0.5:
        st.error('🔴 High risk: customer service should contact this customer.')
    else:
        st.success('🟢 Low risk: no action needed.')

    with st.expander('How to read this score'):
        st.write('The model was trained with balanced class weights, so the score is used '
                 'to rank orders by risk. Orders with a score of 50% or more are flagged.')
