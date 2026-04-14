import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Job Salary Predictor", layout="centered")

# --- LOAD THE TRAINED MODEL AND COLUMNS ---
@st.cache_resource
def load_model():
    model = pickle.load(open('best_salary_model.pkl', 'rb'))
    columns = pickle.load(open('model_columns.pkl', 'rb'))
    return model, columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure .pkl files are in the same folder.")

# --- APP TITLE & DESCRIPTION ---
st.title(" Job Salary Predictor")
st.write("Enter the job details below to estimate the annual salary.")

# --- STEP 1: NUMERICAL INPUTS ---
st.subheader("Professional Details")
col1, col2, col3 = st.columns(3)

with col1:
    exp = st.slider("Years of Experience", 0, 40, 5)
with col2:
    skills = st.number_input("Skills Count", 1, 50, 5)
with col3:
    certs = st.number_input("Certifications", 0, 10, 1)

# --- STEP 2: CATEGORICAL INPUTS ---
st.subheader("Job Profile")
# Note: These options should match your original dataset categories
job_title = st.selectbox("Job Title", ["Software Engineer", "Data Scientist", "Manager", "Analyst", "Developer"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "High School"])
industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Education", "Telecom"])
location = st.selectbox("Location", ["USA", "India", "UK", "Canada", "Singapore"])
remote = st.radio("Remote Work", ["Yes", "No", "Hybrid"])

# --- STEP 3: PREDICTION LOGIC ---
if st.button("Predict Salary"):
    # Create a template dataframe with all zeros matching the model's structure
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill Numerical Values (Exact match to your Excel)
    input_df['experience_years'] = exp
    input_df['skills_count'] = skills
    input_df['certifications'] = certs
    
    # Fill Categorical Values (One-Hot Encoding logic)
    # This finds the correct column (e.g., 'job_title_Data Scientist') and sets it to 1
    for col in model_columns:
        if job_title in col or education in col or industry in col or location in col or remote in col:
            input_df[col] = 1

    # Perform Prediction
    prediction = model.predict(input_df)
    
    # Display Result
    st.success(f"### Estimated Annual Salary: ${prediction[0]:,.2f}")
    st.balloons()

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for Wrexham University Machine Learning Assignment")