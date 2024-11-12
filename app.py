# importing the packages
import streamlit as st
from PIL import Image


# importingthe files
from heart import heart_pred
from cancer import cancer_pred
from diabetes import diabetes_pred
from stroke import stroke_pred
from kidney import kidney_pred

def main():

	menu = ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Chronic Kidney Disease Prediction", "Stroke Prediction",
	"Cancer Prediction", "About"]

	choice = st.sidebar.selectbox("Menu", menu)

	if choice=="Home":

		st.write("# HealthGuard: Web-based Early Detection System ")
		#img = Image.open("cover.jpg")
		#st.image(img)

		st.write("""


HealthGaurd is an advanced healthcare technology that uses machine learning algorithms to predict the likelihood of a patient developing multiple diseases based on their medical history, lifestyle, genetic factors, and other relevant data. The system is designed to help doctors and medical professionals identify patients who are at a higher risk of developing multiple diseases and provide them with early intervention and appropriate medical care to prevent or manage the onset of these diseases. This system can have a significant impact on the healthcare industry by improving patient outcomes, reducing healthcare costs, and enhancing overall population health.


There are a total of 5 diseases that can be predicted with the help of this application.

### App Content

	- This app has 7 sections

	1) Home Page - The page you are currently in

	2) Diabetes Prediction - This page will help you to predict whether you have 
	Diabetes or not

	3) Heart Disease Prediction - This page will help you to predict whether you have 
	Heart disease or not

	4) Chronic Kidney Disease Prediction - This page will help you to predict whether 
	you have Chronic Kidney Disease or not

	5) Liver Disease Prediction - This page will help you to predict whether you have 
	Liver Disease or not

	6) Cancer Prediction - This page will help you to predict whether you have Cancer 
	or not

	7) About - About the Creators

			""")


	elif choice=="Heart Disease Prediction":
		heart_pred()
	elif choice == "Cancer Prediction":
		cancer_pred()
	elif choice == "Diabetes Prediction":
		diabetes_pred()
	elif choice == "Stroke Prediction":
		stroke_pred()
	elif choice == "Chronic Kidney Disease Prediction":
		kidney_pred()
		


main()
