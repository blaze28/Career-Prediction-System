import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import time
import streamlit as st

# Load the model
pickleFile = open("weights.pkl", "rb")
regressor = pickle.load(pickleFile)

# Load the data
df = pd.read_csv("./data/mldata.csv")

# Change column name
df["workshops"] = df["workshops"].replace(["testing"], "Testing")

# Sidebar
st.sidebar.header("User Input Features")
st.sidebar.write(df.columns.tolist())
selected_cols = st.sidebar.multiselect("Select columns", df.columns.tolist())

# Show the data
st.header("Raw Data")
st.write(df[selected_cols])

# Feature Engineering
st.header("Feature Engineering")

# Binary Encoding
st.subheader("Binary Encoding for Categorical Variables")
binary_cols = ["self-learning capability?", "Extra-courses did", "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]
binary_values = {"yes": 1, "no": 0}
df[binary_cols] = df[binary_cols].replace(binary_values)

# Number Encoding
st.subheader("Number Encoding for Categorical Variables")
number_cols = ["reading and writing skills", "memory capability score"]
number_values = {"poor": 0, "medium": 1, "excellent": 2}
df[number_cols] = df[number_cols].replace(number_values)

# Dummy Variable Encoding
st.subheader("Dummy Variable Encoding")
dummy_cols = ["Management or Technical", "hard/smart worker"]
df = pd.get_dummies(df, columns=dummy_cols, prefix=dummy_cols)

# Show the encoded data
st.write(df[selected_cols])

# Show the categorical feature list
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    st.subheader("Categorical Features")
    st.write(categorical_cols)

# Show the numerical feature list
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numerical_cols) > 0:
    st.subheader("Numerical Features")
    st.write(numerical_cols)

# Prediction
st.header("Prediction")

# Collect user input
input_list = []
for col in df.columns.tolist():
    if col in selected_cols:
        if col == "Suggested Job Role":
            continue
        user_input = st.number_input(f"Enter {col}", value=0)
        input_list.append(user_input)

# Make prediction
if st.button("Predict"):
    input_array = np.array(input_list).reshape(1, -1)
    prediction = regressor.predict(input_array)[0]
    st.write(f"The suggested job role is {prediction}")
