import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Define the encoded values for mapping
encoded_values = {
    "Female": 0, "Male": 1, "Yes": 1, "No": 0, "Urban": 1, "Rural": 0,
    'Private': 1, 'Self-employed': 3, 'Govt job': 0, 'Children': 4, 'Never worked': 1,
    'Formerly smoked': 1, 'Never smoked': 2, 'Smokes': 3, 'Unknown': 0
}

# Function for encoding categorical values based on the dictionary
def a(val, my_dict):
    return my_dict.get(val, val)  # Get value from dict or return the original value if not found

# Main prediction function
def stroke_pred():
    st.write("# Stroke Predictor")
    img = Image.open("stroke.jpeg")
    st.image(img)
    st.write("""
    Stroke is a serious medical condition that can lead to long-term disability or death. Early prediction of stroke can help in effective prevention and management of the disease.
    """)
    
    # User input
    st.header("Give Your Input")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("What is your Gender", ["Male", "Female"])
        hypertension = st.radio("Do you have Hypertension?", ["Yes", "No"])
        ever_married = st.radio("Have you ever married before?", ["Yes", "No"])
        residence_type = st.selectbox("Select your Residence Type", ['Rural', 'Urban'])
        bmi = st.number_input("Enter your BMI Value", 10.0, 100.0, 28.0, step=0.1)

    with col2:
        age = st.number_input("Enter your Age", 10, 90, 44)
        heart_disease = st.radio("Do you have Heart Disease?", ["Yes", "No"])
        work_type = st.selectbox("What is your Work type?", ['Private', 'Self-employed', 'Govt job', 'Children', 'Never worked'])
        avg_glucose_level = st.number_input("Enter your Avg glucose level", 55.0, 300.0, 106.0, step=0.1)
        smoking_status = st.selectbox("Select your Smoking Status", ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'])

    # Display selected inputs
    with st.expander("Your selected options"):
        user_input = {
            "Gender": gender, "Age": age, "Hypertension": hypertension, "Heart Disease": heart_disease,
            "Ever Married": ever_married, "Work Type": work_type, "Residence Type": residence_type,
            "Avg Glucose Level": avg_glucose_level, "BMI": bmi, "Smoking Status": smoking_status
        }
        st.write(user_input)

        # Convert user inputs to numerical values
        result = []
        for i in user_input.values():
            if isinstance(i, (int, float)):  # For numeric fields
                result.append(i)
            else:  # For categorical fields
                result.append(a(i, encoded_values))

    # Predict and show results
    with st.expander("Prediction Results"):
        input_data = np.array(result).reshape(1, -1)

        # Load the trained model
        m = joblib.load("stroke_model")

        # Get prediction and probability
        prediction = m.predict(input_data)
        prob = m.predict_proba(input_data)

        # Show prediction result
        if prediction == 1:
            st.warning("Positive Risk!!! You have Stroke, Be Careful")
            prob_score = {"Positive Risk": prob[0][1], "Negative Risk": prob[0][0]}
            st.write(prob_score)
        else:
            st.success("Negative Risk!!! You don't have Stroke, Enjoy!!")
            prob_score = {"Negative Risk": prob[0][0], "Positive Risk": prob[0][1]}
            st.write(prob_score)
