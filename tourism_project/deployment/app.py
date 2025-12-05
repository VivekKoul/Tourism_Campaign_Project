import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="KoulVivek/tourism_project", filename="tourism_project_model_v2.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Campaign and Revenue Prediction")
st.write("""
This application predicts the predicts potential buyers, and enhances decision-making for marketing strategies
Please enter the details below to get a revenue prediction.
""")

# User input  'TypeofContact', 'Occupation', 'Gender', 'ProductPitched','MaritalStatus','Designation'
contact_type = st.selectbox("TypeofContact", ['Self Enquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
gender = st.selectbox("Gender", ['Female', 'Male'])
productpitched = st.selectbox("ProductPitched",['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
maritalstatus = st.selectbox("MaritalStatus", ['Single', 'Divorced', 'Married'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])

age = st.number_input("Age in Yrs)", min_value=1.0, max_value=100.0, value=18.0)
city_tier = st.number_input("Tier)", min_value=1, max_value=3, value=1)
pitch_duration = st.number_input("Duration Of Pitch", min_value=2, max_value=130, value=5)
no_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=5, value=1)
no_of_follow_ups = st.number_input("Number Of Follow ups", min_value=1, max_value=6, value=1)
preferred_property_star = st.number_input("Preferred Property Star", min_value=3, max_value=5, value=3)
no_of_trips = st.number_input("Number Of Trips", min_value=1, max_value=22, value=1)
passport_holder = st.number_input("Passport", min_value=0, max_value=1, value=0)
pitch_satisfaction = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)
car_owner = st.number_input("Own Car", min_value=0, max_value=1, value=0)
no_of_childrn_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=1000)

# Assemble input into DataFrame
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched','MaritalStatus','Designation'
]

input_data = pd.DataFrame ([{
'Age': age,
'CityTier':city_tier ,
'DurationOfPitch':pitch_duration,
'NumberOfPersonVisiting':no_of_person_visiting,
'NumberOfFollowups':no_of_follow_ups,
'PreferredPropertyStar':preferred_property_star,
 'NumberOfTrips':no_of_trips,
'Passport':passport_holder,
'PitchSatisfactionScore':pitch_satisfaction,
'OwnCar':car_owner,
'NumberOfChildrenVisiting':no_of_childrn_visiting,
'MonthlyIncome':MonthlyIncome,
'TypeofContact':contact_type,
'Occupation':occupation ,
'Gender':gender,
'ProductPitched': productpitched,
'MaritalStatus':maritalstatus,
'Designation' :designation
}])

# Predict button
if st.button("Predict Campaign Success"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Probability of Purachasing the package: {prediction:,.2f}")
