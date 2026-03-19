import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data.csv")

# Features & Target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# UI
st.title("🧠 Customer Purchase Prediction App")

st.markdown("Predict whether a customer will purchase a product based on age and salary.")

# User Inputs
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
salary = st.number_input("Enter Salary", min_value=10000, max_value=100000, value=30000)

# Prediction button
if st.button("Predict"):
    input_data = scaler.transform([[age, salary]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Customer is likely to PURCHASE")
    else:
        st.error("❌ Customer is NOT likely to purchase")