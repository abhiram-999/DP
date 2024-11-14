# Import our packages
import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image

attribute = ("""

## Attribute Information

Pregnancies: The number of pregnancies you had.

Glucose: Glucose level in your blood in mg/dL

Blood Pressure: Blood Pressure value mmHg

Skin Thickness: Skin Thickness value in mm

Insulin: Insulin level in blood mmol/L 

BMI: Body Mass Index

Diabetes Pedigree Function: Diabetes Pedigree Value

Age: Age in years


""")


def diabetes_pred():

	st.write("# Diabetes Predictor")
	img = Image.open("diabetes.jpg")
	st.image(img) 
	st.write("""



The Diabetes Predictor App is a powerful tool designed to help individuals assess their risk of developing diabetes. With the increasing prevalence of diabetes around the world, early detection and intervention are critical in managing the condition and reducing its complications. The app utilizes machine learning algorithms and predictive models trained on large datasets to analyze multiple risk factors, such as age, BMI, blood pressure, and glucose levels, to generate an accurate prediction of an individual's likelihood of developing diabetes. 

By providing users with personalized risk assessments, the Diabetes Predictor App empowers you to have an idea on whether you are at risk for diabetes or simply looking to maintain a healthy lifestyle, the Diabetes Predictor App is an essential tool for optimizing your health and wellness.
		""")

	st.markdown(attribute)

	st.header("Give Your Input")
 

	col1, col2 = st.columns(2)

	with col1:
		pregnancies = st.number_input("Enter the number of Pregnancies you had", 0, 50, 2)
		blood_pressure = st.number_input("Enter your Blood Pressure", 0, 360, 80)
		insulin = st.number_input("Enter your Insulin Level", 0, 1000, 100)
		diabetes_pedigree_function = st.number_input("Enter your Diabetes Pedigree Function Value", 0.0, 5.0, 0.05, step = 0.001)
		

	with col2:
		glucose = st.number_input("Enter your Glucose Level", 0, 210, 130)
		skin_thickness = st.number_input("Enter the Thickness of your skin", 0, 200, 20)
		bmi = st.number_input("Enter your BMI value", 15.0, 100.0, 25.0, step = 0.1)
		age = st.number_input("Enter your Age", 12, 100, 20)


	with st.expander("Your selected options"):
		so = {"Pregnancies":pregnancies, "Glucose":glucose, "BloodPressure":blood_pressure, "SkinThickness":skin_thickness,
		"Insulin":insulin, "BMI":bmi, "DiabetesPedigreeFunction":diabetes_pedigree_function, "Age":age}

		st.write(so)


		result = []

		for i in so.values():
			if type(i) == int or type(i) == float:
				result.append(i)

	with st.expander("Prediction Results"):

		input = np.array(result).reshape(1,-1)
		#st.write(input)

		m = joblib.load("diabetes_model")

		prediction = m.predict(input)
		#st.write(prediction)

		prob = m.predict_proba(input)
		#st.write(prob)

		if prediction == 1:
			st.warning("Positive Risk!!!, You have Diabetes")
			prob_score = {"Positive Risk": prob[0][1],
			"Negative Risk": prob[0][0]}
			st.write(prob_score)

		else:
			st.success("Negative Risk!!!, You don't have Diabetes")
			prob_score = {"Negative Risk": prob[0][0],
			"Positive Risk": prob[0][1]}
			st.write(prob_score)
