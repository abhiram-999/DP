# Import necessary libraries
import streamlit as st
import numpy as np
from PIL import Image
from xgboost import XGBClassifier

# Predefined attribute information and encoding dictionary
attribute = """
## Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt job", "Never worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
"""

encoded_values = {
    "Female": 0, "Male": 1, "Yes": 1, "No": 0, "Urban": 1, "Rural": 0, 'Private': 1,
    'Self-employed': 3, 'Govt job': 0, 'Children': 4, 'Never worked': 1, 'Formerly smoked': 1,
    'Never smoked': 2, 'Smokes': 3, 'Unknown': 0
}

# Function for encoding
def encode(val, encoding_dict):
    return encoding_dict.get(val, val)

# Stroke prediction function
def stroke_pred():
    st.write("# Stroke Predictor")
    img = Image.open("stroke.jpeg")
    st.image(img) 
    st.write(attribute)

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

    with st.expander("Your selected options"):
        input_data = {"Gender": gender, "Age": age, "Hypertension": hypertension, "Heart Disease": heart_disease,
                      "Ever Married": ever_married, "Work Type": work_type, "Residence Type": residence_type,
                      "Avg Glucose Level": avg_glucose_level, "BMI": bmi, "Smoking Status": smoking_status}
        st.write(input_data)

        result = []
        for i in input_data.values():
            if isinstance(i, (int, float)):
                result.append(i)
            else:
                result.append(encode(i, encoded_values))

    with st.expander("Prediction Results"):
        input_array = np.array(result).reshape(1, -1)

        # Load the pre-trained model in JSON format
        model = XGBClassifier()
        model.load_model("stroke_model.json")

        # Make predictions
        prediction = model.predict(input_array)
        prob = model.predict_proba(input_array)

        if prediction == 1:
            st.warning("Positive Risk! You may have a stroke")
            prob_score = {"Positive Risk": prob[0][1], "Negative Risk": prob[0][0]}
            st.write(prob_score)
        else:
            st.success("Negative Risk! You don't have a stroke")
            prob_score = {"Negative Risk": prob[0][0], "Positive Risk": prob[0][1]}
            st.write(prob_score)

# Run the Streamlit app
if __name__ == "__main__":
    stroke_pred()
