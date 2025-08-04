# churn_ui.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained models (you can save them using joblib or pickle)
# Example: joblib.dump(lr, 'logistic_model.pkl') after training
lr_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

st.title("Customer Churn Prediction App")

st.markdown("Upload a CSV file with customer data (same structure as training data)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Encode 'Churn' if present (optional)
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # Get dummies for categorical columns
    data = pd.get_dummies(data, drop_first=True)

    # Align columns with training data (optional, handle mismatch)
    # You must ensure the same feature structure as the training data

    st.write("### Preview of Uploaded Data")
    st.dataframe(data)

    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

    if st.button("Predict Churn"):
        if 'Churn' in data.columns:
            X = data.drop('Churn', axis=1)
        else:
            X = data

        if model_choice == "Logistic Regression":
            predictions = lr_model.predict(X)
        else:
            predictions = rf_model.predict(X)

        data['Churn Prediction'] = predictions
        st.write("### Prediction Results")
        st.dataframe(data)

        st.download_button("Download Prediction CSV", data.to_csv(index=False), file_name="churn_predictions.csv", mime="text/csv")
