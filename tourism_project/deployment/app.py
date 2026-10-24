import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing the wellness tourism package.
""")

age = st.number_input("Age", min_value=18, max_value=100, step=1, value=25)
type_of_contact = st.selectbox("Type Of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.number_input("City Tier", min_value=1, max_value=3, step=1, value=1)
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1.0, max_value=60.0,step=0.5, value=5.0)
occupation = st.selectbox("Occupation",["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1, max_value=20, step=1, value=2)
number_of_followups = st.number_input("Number of Followups", min_value=1, max_value=10,step=1, value=2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.number_input("Preferred Property Rating", min_value=1, max_value=5, step=1, value=5)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=100, step=1, value=2)
passport = st.number_input("Passport", min_value=0, max_value=1, step=1, value=0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5,step=1, value=5)
own_car = st.number_input("Own Car", min_value=0, max_value=1, step=1, value=0)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10,step=1, value=2)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=1000000.0,step=100.0, value=30000.0)


input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer buys the wellness package" if prediction == 1 else "Customer does not buy the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
