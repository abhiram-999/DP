import streamlit as st
import cloudpickle
import numpy as np
from PIL import Image

# Attribute Information
attribute = ("""

## Attribute Information

age: age in years

sex: sex (1 = male; 0 = female)

cp: chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic

trestbps: resting blood pressure (in mm Hg on admission to the hospital)

chol: serum cholestoral in mg/dl

fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach: maximum heart rate achieved

exang: exercise induced angina (1 = yes; 0 = no)

oldpeak = ST depression induced by exercise relative to rest

 slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping

ca: number of major vessels (0-3) colored by flourosopy

thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

num: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing

""")

# Encoding dictionary
encoded_values = {
    "Female": 0, "Male": 1, 'typical angina': 0, 'atypical angina': 1,
    "non-anginal pain": 2, "asymptomatic": 3, "lower than 120mg/ml": 0,
    "greater than 120mg/ml": 1, 'normal': 0, 'ST-T wave abnormality': 1, 
    'left ventricular hypertrophy': 2, "no": 0, "yes": 1, 
    "upsloping": 0, "flat": 1, "downsloping": 2, "normal": 1, 
    "fixed defect": 2, "reversable defect": 3
}

# Encoding function
def encode_value(val, my_dict):
    return my_dict.get(val, val)

def heart_pred():
    st.write("# Heart Disease Predictor")
    img = Image.open("heart.jpeg")
    st.image(img)

    st.markdown(attribute)
    st.header("Give Your Input")

    # Collect inputs
    age = st.number_input("Enter your Age", 29, 80, 54)
    sex = st.radio("What is your Gender", ["Male", "Female"])
    chest_pain_type = st.selectbox("Your Chest pain type", ['typical angina', 'atypical angina', "non-anginal pain", "asymptomatic"])
    resting_blood_pressure = st.number_input("Resting Blood Pressure value", 90, 200, 131)
    cholestrol = st.number_input("Cholesterol Value", 120, 575, 245)
    fasting_blood_sugar = st.radio("Your Fasting blood sugar value is", ["lower than 120mg/ml", 'greater than 120mg/ml'])
    rest_ecg = st.selectbox("Your Rest ECG type", ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
    max_heart_rate_achieved = st.number_input("Your maximum heart rate achieved value", 70, 210, 109)
    exercise_induced_angina = st.radio("Exercise Induced Angina", ["yes", "no"])
    st_depression = st.number_input("Your ST depression value", 0.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("Your ST Slope type", ['upsloping', 'flat', 'downsloping'])
    num_major_vessels = st.number_input("How many major vessels do you have?", 0, 3, 1)
    thalassemia = st.selectbox("Your Thalassemia type", ["normal", "fixed defect", "reversable defect"])

    # Encode and prepare input data
    inputs = [
        age, encode_value(sex, encoded_values), encode_value(chest_pain_type, encoded_values),
        resting_blood_pressure, cholestrol, encode_value(fasting_blood_sugar, encoded_values),
        encode_value(rest_ecg, encoded_values), max_heart_rate_achieved,
        encode_value(exercise_induced_angina, encoded_values), st_depression,
        encode_value(st_slope, encoded_values), num_major_vessels, encode_value(thalassemia, encoded_values)
    ]
    input_data = np.array(inputs).reshape(1, -1)

    # Load the model using cloudpickle
    try:
        with open('heart_model.pkl', 'rb') as f:
            model = cloudpickle.load(f)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Make predictions
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    if prediction == 1:
        st.warning("Positive Risk!! You have Heart Disease, Be Careful")
        st.write({"Positive Risk": prob[0][1], "Negative Risk": prob[0][0]})
    else:
        st.success("Negative Risk! You don't have Heart Disease.")
        st.write({"Negative Risk": prob[0][0], "Positive Risk": prob[0][1]})

# Run the prediction function
if __name__ == "__main__":
    heart_pred()
