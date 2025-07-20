import streamlit as st
import pandas as pd
import joblib
# Load model
model= joblib.load("Gradient Boosting_best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification",page_icon="🎯",layout="centered")
st.title("🎯EMPLOYEE SALARY CLASSIFICATION APP")
st.markdown("Predict whether an Employee earns >50k or <=50k based on input features.")

# Sidebar inputs(these must match your training features columns)
st.sidebar.header("Input Employee details")

# Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age",18,65,30)
education = st.sidebar.selectbox("Education Level",["Bachelors","Masters","PHD","MS Grad","Assoc","Some college"])
occupation = st.sidebar.selectbox("Jobrole",["Tech support","Craft repair","Other services","Sales","Handlers-cleaners","Transport moving","Priv-house-serve","Protective-serve","Armed forces"])
hours_per_week = st.sidebar.slider("Hours per week",1,89,40)
experience = st.sidebar.slider("Years of Experience",0,40,5)

# Build input dataframes(must watch processing of your input data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours_per_week': [hours_per_week],
    'experience': [experience]
})
st.write("Input data")
st.write(input_df)

# Predict button
if st.button("Predict salary class"):
    prediction = model.predict(input_df)
    st.success(f"Prediction:{prediction[0]}")

# Batch prediction
st.markdown("___")
st.markdown("Batch prediction")
uploaded_file = st.file_uploader("upload a csv file for batch prediction",type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("uploaded data preview:",batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data["Predicted class"] = batch_preds
    st.write("Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV",csv,file_name='predicted_classes.csv',mime='text/csv')
