
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
	"Cancer Prediction"]

	choice = st.sidebar.selectbox("Menu", menu)

	if choice=="Home":

		st.write("# HealthGuard: Web-based Early Detection System ")
		#img = Image.open("cover.jpg")
		#st.image(img)

		# Introduction text
		st.write("""
        ### Welcome to HealthGuard!
        HealthGuard uses advanced machine learning algorithms to predict the likelihood of a patient developing multiple diseases. It’s designed to support early intervention based on their medical history, lifestyle, genetic factors, and other relevant data and appropriate medical care to improve patient outcomes.
        
        ### Diseases Predicted
        - *Diabetes*
        - *Heart Disease*
        - *Chronic Kidney Disease*
        - *Stroke*
        - *Cancer*

        Navigate through the sections to predict your health risk for each disease.
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
