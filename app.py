import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the Titanic dataset
path = "/Users/sandrathermildor/.cache/kagglehub/datasets/dwiuzila/titanic-machine-learning-from-disaster/versions/2"

# Try to load the datasets and handle errors
try:
    train_data = pd.read_csv(f"{path}/train.csv")
    st.success("✅ Train dataset loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

# App title
st.title("Titanic Survival Prediction App")

# Checkbox to display raw data
if st.checkbox("Show raw data"):
    st.subheader("Train Dataset")
    st.write(train_data.head())

# Plot Age Distribution
st.subheader("Age Distribution of Passengers")
plt.figure(figsize=(10, 5))
plt.hist(train_data['Age'].dropna(), bins=20, edgecolor='black')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution in Train Dataset")
st.pyplot(plt)

# Summary statistics
st.subheader("Summary Statistics")
st.write(train_data.describe())

# Select Passenger Class to filter data
pclass = st.selectbox("Select Passenger Class (Pclass):", [1, 2, 3])
filtered_data = train_data[train_data['Pclass'] == pclass]

# Display filtered data
st.subheader(f"Passengers in Class {pclass}")
st.write(filtered_data)

# Display the survival rate by class
st.subheader("Survival Rate by Passenger Class")
survival_rate = train_data.groupby('Pclass')['Survived'].mean()
st.bar_chart(survival_rate)