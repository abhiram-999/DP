import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image

def cancer_pred():

    st.write("# Cancer Predictor")
    img = Image.open("cancer.jpeg")
    st.image(img) 
    st.write("""

Cancer is a complex disease that affects millions of people around the world, making it a major public health concern. 
Early detection of cancer can significantly improve a patient's prognosis, and machine learning has shown great promise in predicting the likelihood of developing cancer. 
We can use this webpage as a tool that uses machine learning algorithms to analyze a patient's health data and predict their risk of developing cancer. 
The app takes into account various risk factors such as size, smoothness, radius and many other parameters to provide personalized cancer risk assessments. 
By using this app, patients can take proactive measures to prevent cancer or detect it early, ultimately leading to better health outcomes.
    
            """)

    st.header("Give Your Input")
 

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        radius_mean = st.number_input("Radius Mean", 6.0, 30.0, 15.0, step = 0.01)
        perimeter_mean = st.number_input("Perimeter Mean", 40.0, 200.0, 92.0, step = 0.01)
        smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.2, step = 0.01)
        concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.3, step = 0.01)
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2, step = 0.01)
        radius_se = st.number_input("Radius SE", 0.0, 5.0, 0.5 ,step = 0.01)
        
        
        

    with col2:
        texture_mean = st.number_input("Texture Mean", 9.0, 45.0, 20.0, step = 0.01)
        area_mean = st.number_input("Area Mean", 0.0, 3000.0, 650.0, step = 0.1)
        compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.2, step = 0.01)
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.2, step = 0.01)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.4, step = 0.01)
        texture_se = st.number_input("Texture SE", 0.0, 10.0, 3.0, step = 0.01)
        
        

    with col3:
        radius_worst = st.number_input("Radius Worst", 5.0, 50.0, 25.0, step = 0.01)
        perimeter_worst = st.number_input("Perimeter Worst", 30.0, 300.0, 70.0, step = 0.01)
        smoothness_worst = st.number_input("Smoothness Worst", 0.0, 1.0, 0.3, step = 0.01)
        concavity_worst = st.number_input("Concavity Worst", 0.0, 3.0, 1.0, step = 0.01)
        symmetry_worst = st.number_input("Symmetry Worst", 0.0, 1.0, 0.2, step = 0.01)
        texture_worst = st.number_input("Texture Worst", 0.0, 100.0, 30.0, step = 0.01)
        

    with col4:
        compactness_se = st.number_input("Compactness SE", 0.0, 1.0, 0.3, step = 0.01)
        concave_points_se = st.number_input("Concave Points SE", 0.0, 1.0, 0.4, step = 0.01)
        fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, 1.0, 0.5, step = 0.001)
        compactness_worst = st.number_input("Compactness Worst", 0.0, 2.0, 0.6, step = 0.01)
        concave_points_worst = st.number_input("Concave Points Worst", 0.0, 1.0, 0.5, step = 0.01)
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, 1.0, 0.5, step = 0.01)
        
    with col5:
        smoothness_se = st.number_input("Smootheness SE", 0.0, 1.0, 0.2, step = 0.01)
        concavity_se = st.number_input("Concavity SE", 0.0, 1.0, 0.2, step = 0.01)
        perimeter_se = st.number_input("Perimeter SE", 0.0, 50.0, 5.0, step = 0.01)
        area_se = st.number_input("Area SE", 0.0, 1000.0, 200.0, step = 0.01)
        area_worst = st.number_input("Area Worst", 0.0, 5000.0, 500.0, step = 0.01)
        symmetry_se = st.number_input("Symmetry SE", 0.0, 1.0, 0.2, step = 0.01)




    with st.expander("Your selected options"):
        so = {"radius_mean":radius_mean, "texture_mean": texture_mean, "perimeter_mean": perimeter_mean, "area_mean": area_mean,
        "smoothness_mean": smoothness_mean, "compactness_mean": compactness_mean, "concavity_mean": concavity_mean, 
        "concave points_mean": concave_points_mean, "symmetry_mean":symmetry_mean,"fractal_dimension_mean":fractal_dimension_mean,

        "radius_se":radius_se, "texture_se": texture_se, "perimeter_se": perimeter_se, "area_se": area_se,
        "smoothness_se": smoothness_se, "compactness_se": compactness_se, "concavity_se": concavity_se, 
        "concave points_se": concave_points_se, "symmetry_se":symmetry_se,"fractal_dimension_se":fractal_dimension_se,

        "radius_worst":radius_worst, "texture_worst": texture_worst, "perimeter_worst": perimeter_worst, "area_worst": area_worst,
        "smoothness_worst": smoothness_worst, "compactness_worst": compactness_worst, "concavity_worst": concavity_worst, 
        "concave points_worst": concave_points_worst, "symmetry_worst":symmetry_worst,"fractal_dimension_worst":fractal_dimension_worst}

        st.write(so)


        result = []

        for i in so.values():
            if type(i) == int or type(i) == float:
                result.append(i)

        #st.write(result)

    with st.expander("Prediction Results"):

        input = np.array(result).reshape(1,-1)
        #st.write(input)

        m = joblib.load("cancer_model")

        prediction = m.predict(input)
        #sst.write(prediction)

        prob = m.predict_proba(input)
        #st.write(prob)

        if prediction == 1:
            st.warning("Positive Risk!!!")
            prob_score = {"Positive Risk": prob[0][1],
            "Negative Risk": prob[0][0]}
            st.write(prob_score)

        else:
            st.success("Negative Risk!!!")
            prob_score = {"Negative Risk": prob[0][0],
            "Positive Risk": prob[0][1]}
            st.write(prob_score)


     


