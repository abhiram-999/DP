# Import our packages
import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image


attribute = ("""

## Attribute Information

1.Age(numerical)
age in years

2.Blood Pressure(numerical)
bp in mm/Hg

3.Specific Gravity(nominal)
sg - (1.005,1.010,1.015,1.020,1.025)

4.Albumin(nominal)
al - (0,1,2,3,4,5)

5.Sugar(nominal)
su - (0,1,2,3,4,5)

6.Red Blood Cells(nominal)
rbc - (normal,abnormal)

7.Pus Cell (nominal)
pc - (normal,abnormal)

8.Pus Cell clumps(nominal)
pcc - (present,notpresent)

9.Bacteria(nominal)
ba - (present,notpresent)

10.Blood Glucose Random(numerical)
bgr in mgs/dl

11.Blood Urea(numerical)
bu in mgs/dl

12.Serum Creatinine(numerical)
sc in mgs/dl

13.Sodium(numerical)
sod in mEq/L

14.Potassium(numerical)
pot in mEq/L

15.Hemoglobin(numerical)
hemo in gms

16.Packed Cell Volume(numerical)

17.White Blood Cell Count(numerical)
wc in cells/cumm

18.Red Blood Cell Count(numerical)
rc in millions/cmm

19.Hypertension(nominal)
htn - (yes,no)

20.Diabetes Mellitus(nominal)
dm - (yes,no)

21.Coronary Artery Disease(nominal)
cad - (yes,no)

22.Appetite(nominal)
appet - (good,poor)

23.Pedal Edema(nominal)
pe - (yes,no)

24.Anemia(nominal)
ane - (yes,no)

25.Class (nominal)
class - (ckd,notckd)


""")


encoded_values = {"Normal":1, "Abnormal":0, 
"NORMAL":0, "ABNORMAL":1,
"Not Present":0, "Present":1, 
"NOT PRESENT":0, "PRESENT":1,
"Yes":1, "No":0, "Good":0, "Poor":1}

# Function for encoding
def a(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

def kidney_pred():

	st.write("# Chronic Kidney Disease Predictor")
	img = Image.open("kidney.jpeg")
	st.image(img) 
	st.write("""

Chronic kidney disease (CKD) is a serious condition that affects millions of people worldwide. Early detection and management of CKD can improve outcomes and prevent progression to end-stage renal disease. To address this need, we have developed a Chronic Kidney Disease Predictor App that uses machine learning algorithms to predict the risk of CKD in patients based on various clinical and demographic factors.

Our app takes into account important risk factors for CKD, such as age, gender, blood pressure, blood sugar, albumin and many other important factors. By inputting these factors into the app, patients can receive a personalized risk assessment and take proactive steps to prevent or manage CKD.

The app is easy to use and provides patients with an intuitive interface that displays their risk score which can be uselful and let the user know whther they have CKD or Not. With early detection and intervention, we hope to reduce the burden of CKD on patients and healthcare systems.
		""")

	st.markdown(attribute)

	st.header("Give Your Input")
 

	col1, col2, col3 = st.columns(3)

	with col1:
		age = st.number_input("Enter your Age", 10, 100, 40)
		specific_gravity = st.number_input("Enter your Specific Gravity value", 1.0, 1.03, 1.015, step = 0.01)
		sugar = st.number_input("Enter your Sugar Level", 0, 5, 3)
		pus_cell = st.radio("Enter your Pus Cell Level", ["Normal", "Abnormal"])
		bacteria = st.radio("Do you have Bacteria?", ["PRESENT", "NOT PRESENT"])
		blood_urea = st.number_input("Enter your Blood Urea Value", 0.0, 500.0, 60.0, step = 0.1)
		sodium = st.number_input("Enter your Sodium Level", 1.0, 250.0, 130.0, step = 0.1)
		haemoglobin = st.number_input("Enter your Haemoglobin Level", 0.0, 25.0, 12.0, step = 0.1)
		

	with col2:
		blood_pressure = st.number_input("Enter your Blood Pressure Level", 40, 200, 75)
		albumin = st.number_input("Enter your Albumin Level", 0, 5, 3)
		red_blood_cells = st.selectbox("What is your Red Blood Cell Level?", ["NORMAL", "ABNORMAL"])
		pus_cell_clumps = st.radio("Do you have Pus Cell Clumps?", ["Present", "Not Present"])
		blood_glucose_random = st.number_input("What is your Random Blood Glucose Level", 0, 500, 150)
		serum_creatinine = st.number_input("Enter your Serum Creatinine Value", 0.1, 100.0, 5.0, step = 0.1)
		potassium = st.number_input("Enter your Potassium Level", 1.0, 100.0, 50.0, step = 0.1)
		packed_cell_volume = st.number_input("Enter your Packed Cell Volume Value", 0, 100, 50)
		

	with col3:
		white_blood_cell_count = st.number_input("Enter your White Blood Cell Count", 1000, 50000, 7500)
		hypertension = st.selectbox("Do you have Hypertension?", ["Yes", "No"])
		coronary_artery_disease = st.radio("Do you have Coronary Artery Disease", ["Yes", "No"])
		peda_edema = st.selectbox("Do you have Pedal Edema", ["Yes", "No"])
		red_blood_cell_count = st.number_input("Enter your Red Blood Cell Count", 0.0, 20.0, 5.0, step = 0.1)
		diabetes_mellitus = st.radio("Do you have Diabetes Mellitus?", ["Yes", "No"])
		appetite = st.selectbox("How is your Appetite?", ["Poor", "Good"])
		aanemia = st.radio("Do you have Anemia?", ["Yes", "No"])


	with st.expander("Your selected options"):
		so = {"age":age, "blood_pressure":blood_pressure,"specific_gravity":specific_gravity, "albumin":albumin,"sugar":sugar,
		"red_blood_cells":red_blood_cells, "pus_cell":pus_cell, "pus_cell_clumps":pus_cell_clumps, "bacteria":bacteria,
		"blood_glucose_random":blood_glucose_random, "blood_urea":blood_urea, "serum_creatinine":serum_creatinine, "sodium":sodium,
		"potassium":potassium, "haemoglobin":haemoglobin, "packed_cell_volume":packed_cell_volume, "white_blood_cell_count":white_blood_cell_count,
		"red_blood_cell_count":red_blood_cell_count, "hypertension":hypertension, "diabetes_mellitus":diabetes_mellitus, 
		"coronary_artery_disease":coronary_artery_disease, "appetite":appetite, "peda_edema":peda_edema, "aanemia":aanemia}

		st.write(so)


		result = []

		for i in so.values():
			if type(i) == int or type(i) == float:
				result.append(i)

			else:
				res = a(i, encoded_values)
				result.append(res)

		#st.write(result)

	with st.expander("Prediction Results"):

		input = np.array(result).reshape(1,-1)
		#st.write(input)

		m = joblib.load("kidney_model")

		prediction = m.predict(input)
		#st.write(prediction)

		prob = m.predict_proba(input)
		#st.write(prob)

		if prediction == 0:
			st.warning("Positive Risk!!! You have CKD")
			prob_score = {"Positive Risk": prob[0][0],
			"Negative Risk": prob[0][1]}
			st.write(prob_score)

		else:
			st.success("Negative Risk!!! You don't have CKD")
			prob_score = {"Negative Risk": prob[0][1],
			"Positive Risk": prob[0][0]}
			st.write(prob_score)
