# Import necessary packages
import streamlit as st
import cloudpickle
import numpy as np
from PIL import Image

# Placeholder class to handle any missing attributes
class EuclideanDistance64:
    pass

def load_model():
    try:
        with open("heart_model.pkl", "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Attribute information and encoding dictionary
attribute = """
## Attribute Information
... (details of attributes as in your previous code)
"""

encoded_values = {
    "Female":0, "Male":1, 'typical angina':0, 'atypical angina':1,
    "non-anginal pain":2, "asymptomatic":3, "lower than 120mg/ml":0,
    "greater than 120mg/ml":1, 'normal':0, 'ST-T wave abnormality':1,
    'left ventricular hypertrophy':2, "no":0, "yes":1,
    "upsloping":0, "flat":1, "downsloping":2, "normal":1, 
    "fixed defect":2, "reversable defect":3
}

def encode_input(value, encoding_dict):
    return encoding_dict.get(value, value)

def heart_pred():
    st.write("# Heart Disease Predictor")
    img = Image.open("heart.jpeg")
    st.image(img)
    
    # Attribute description
    st.markdown(attribute)
    st.header("Provide Your Input")

    col1, col2 = st.columns(2)
    # Collect user inputs
    with col1:
        age = st.number_input("Age", 29, 80, 54)
        chest_pain_type = st.selectbox("Chest pain type", 
                                       ['typical angina', 'atypical angina', "non-anginal pain", "asymptomatic"])
        cholestrol = st.number_input("Cholesterol", 120, 575, 245)
        rest_ecg = st.selectbox("Rest ECG type", 
                                ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
        exercise_induced_angina = st.radio("Exercise Induced Angina", ["yes", "no"])
        st_slope = st.selectbox("ST Slope type", ['upsloping', 'flat', 'downsloping'])

    with col2:
        sex = st.radio("Gender", ["Male", "Female"])
        resting_blood_pressure = st.number_input("Resting Blood Pressure", 90, 200, 131)
        fasting_blood_sugar = st.radio("Fasting Blood Sugar", ["lower than 120mg/ml", 'greater than 120mg/ml'])
        max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved", 70, 210, 109)
        st_depression = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)
        num_major_vessels = st.number_input("Number of Major Vessels", 0, 3, 1)
        thalassemia = st.selectbox("Thalassemia type", ["normal", "fixed defect", "reversable defect"])

    # Display user choices
    with st.expander("Selected Options"):
        input_data = [
            age, encode_input(sex, encoded_values),
            encode_input(chest_pain_type, encoded_values),
            resting_blood_pressure, cholestrol, 
            encode_input(fasting_blood_sugar, encoded_values), 
            encode_input(rest_ecg, encoded_values), max_heart_rate_achieved, 
            encode_input(exercise_induced_angina, encoded_values), st_depression, 
            encode_input(st_slope, encoded_values), num_major_vessels, 
            encode_input(thalassemia, encoded_values)
        ]
        st.write(input_data)

    # Load and predict
    with st.expander("Prediction Results"):
        # Load model with cloudpickle and custom objects
        def load_model():
            try:
                with open("heart_model.pkl", "rb") as f:
                    return cloudpickle.load(f)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

        model = load_model()
        if model:
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)
            prob = model.predict_proba(input_array)

            if prediction[0] == 1:
                st.warning("Positive Risk: You may have Heart Disease.")
                st.write({"Positive Risk": prob[0][1], "Negative Risk": prob[0][0]})
            else:
                st.success("Negative Risk: You seem free of Heart Disease.")
                st.write({"Negative Risk": prob[0][0], "Positive Risk": prob[0][1]})

if __name__ == "__main__":
    heart_pred()
