import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Wellness Predictor", layout="centered")
st.title("🧘 Tourism Wellness Package – Purchase Propensity")

@st.cache_resource
def load_model():
    return joblib.load("best_pipeline.joblib")

model = load_model()

st.markdown("Fill the form and click **Predict**.")

with st.form("form"):
    Age = st.number_input("Age", 18, 100, 34)
    CityTier = st.selectbox("CityTier", [1,2,3], index=0)
    MonthlyIncome = st.number_input("MonthlyIncome", 0, 500000, 80000)
    DurationOfPitch = st.number_input("DurationOfPitch (minutes)", 0, 120, 20)
    NumberOfTrips = st.number_input("NumberOfTrips (per year)", 0, 40, 4)
    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", 0, 10, 1)
    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", 1, 20, 2)
    PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [1,2,3,4,5], index=3)
    PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1,2,3,4,5], index=3)
    NumberOfFollowups = st.number_input("NumberOfFollowups", 0, 20, 2)
    TypeofContact = st.selectbox("TypeofContact", ["Company Invited","Self Enquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
    Gender = st.selectbox("Gender", ["Male","Female"])
    MaritalStatus = st.selectbox("MaritalStatus", ["Single","Married","Divorced"])
    ProductPitched = st.selectbox("ProductPitched", ["Basic","Standard","Deluxe","King"])
    Designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])
    Passport = st.selectbox("Passport", [0,1], index=1)
    OwnCar = st.selectbox("OwnCar", [0,1], index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{
        "Age": Age, "CityTier": CityTier, "MonthlyIncome": MonthlyIncome, "DurationOfPitch": DurationOfPitch,
        "NumberOfTrips": NumberOfTrips, "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "NumberOfPersonVisiting": NumberOfPersonVisiting, "PreferredPropertyStar": PreferredPropertyStar,
        "PitchSatisfactionScore": PitchSatisfactionScore, "NumberOfFollowups": NumberOfFollowups,
        "TypeofContact": TypeofContact, "Occupation": Occupation, "Gender": Gender,
        "MaritalStatus": MaritalStatus, "ProductPitched": ProductPitched, "Designation": Designation,
        "Passport": Passport, "OwnCar": OwnCar
    }])
    proba = model.predict_proba(df)[:,1][0]
    pred = int(proba >= 0.5)
    st.metric("Purchase Propensity", f"{proba:.2%}")
    st.success("Likely to purchase ✅" if pred==1 else "Unlikely to purchase ❌")
