import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load('rf_cart_abandonment_model.pkl')

# Streamlit app
st.title("🛒 Cart Abandonment Prediction")

st.write("""
### Enter customer session data:
""")

# Input fields
No_Checkout_Confirmed = st.number_input('Number of checkouts Confirmed', min_value=0, step=1)
No_Customer_Login = st.number_input('Number of times the customer logged in', min_value=0, step=1)
Session_Activity_Count = st.number_input('Number of pages visited', min_value=0, step=1)
No_Checkout_Initiated = st.number_input('Number of checkouts initiated', min_value=0, step=1)
Is_Product_Details_viewed = st.selectbox('Is Product Details Viewed?', [0, 1])

# Predict button
if st.button('Predict'):
    # Prepare features
    features = np.array(
        [[Session_Activity_Count, No_Customer_Login, No_Checkout_Confirmed, No_Checkout_Initiated, Is_Product_Details_viewed]])

    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1][0]

    # Display results
    if prediction[0] == 1:
        st.error(f"⚠️ Prediction: Cart Abandoned (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Prediction: Cart Completed (Probability: {probability:.2%})")
