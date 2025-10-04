import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tourism Wellness Predictor", page_icon="🧭", layout="centered")
st.title("🧭 Tourism Wellness Package — Purchase Prediction")

# ---------------------------------------------------------
# 1) Load trained pipeline (works in Space & locally)
# ---------------------------------------------------------
# We upload best_pipeline.joblib into the Space root.
CANDIDATE_PATHS = ["best_pipeline.joblib", "models/best_pipeline.joblib"]
MODEL_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

if MODEL_PATH is None:
    st.error("❌ best_pipeline.joblib not found. Ensure your CI uploaded it to the Space.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.caption(f"Loaded model: `{MODEL_PATH}`")

# ---------------------------------------------------------
# 2) Define schema used during training
#    (must match the columns your model expects)
# ---------------------------------------------------------
CAT_COLS = [
    "TypeofContact","Occupation","Gender","MaritalStatus",
    "ProductPitched","Designation"
]
NUM_COLS = [
    "Age","CityTier","DurationOfPitch","NumberOfTrips","NumberOfFollowups",
    "PreferredPropertyStar","NumberOfPersonVisiting","NumberOfChildrenVisiting",
    "MonthlyIncome","PitchSatisfactionScore","Passport","OwnCar"
]
ALL_COLS = CAT_COLS + NUM_COLS

# Defaults for the "extra" features often missing in quick demos
DEFAULTS = {
    "CityTier": 2,
    "Passport": 0,
    "OwnCar": 0,
    "PitchSatisfactionScore": 3,
}

# ---------------------------------------------------------
# 3) UI — key fields first; others under Advanced
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=110, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Freelancer","Others"])
    Gender = st.selectbox("Gender", ["Male","Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic","Standard","Deluxe","Super Deluxe","King"])

with col2:
    DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=600, value=30)
    NumberOfTrips = st.number_input("Number Of Trips (avg/yr)", min_value=0, max_value=50, value=2)
    NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=50, value=2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1,2,3,4,5], index=2)
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=20, value=2)
    NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (<5 yrs)", min_value=0, max_value=10, value=0)

MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=2_000_000, value=50_000, step=1000)

with st.expander("Advanced options (optional)"):
    CityTier = st.selectbox("City Tier", [1,2,3], index=DEFAULTS["CityTier"]-1)
    Passport = st.selectbox("Passport (0/1)", [0,1], index=DEFAULTS["Passport"])
    OwnCar = st.selectbox("Own Car (0/1)", [0,1], index=DEFAULTS["OwnCar"])
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1,2,3,4,5], index=DEFAULTS["PitchSatisfactionScore"]-1)

# ---------------------------------------------------------
# 4) Build single-row DataFrame with all required columns
# ---------------------------------------------------------
row = {
    # categoricals (strings)
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Designation": st.selectbox("Designation", ["AVP","VP","Manager","Senior Manager","Executive"], index=4),

    # numerics
    "Age": Age,
    "CityTier": locals().get("CityTier", DEFAULTS["CityTier"]),
    "DurationOfPitch": DurationOfPitch,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": locals().get("PitchSatisfactionScore", DEFAULTS["PitchSatisfactionScore"]),
    "Passport": locals().get("Passport", DEFAULTS["Passport"]),
    "OwnCar": locals().get("OwnCar", DEFAULTS["OwnCar"]),
}

# Ensure every expected column is present exactly once
df = pd.DataFrame([{c: row[c] for c in ALL_COLS}])

# Enforce dtypes expected by the training pipeline
for c in CAT_COLS:
    df[c] = df[c].astype("string")
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

st.write("### Preview of the row sent to the model")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# 5) Predict
# ---------------------------------------------------------
if st.button("Predict"):
    try:
        proba = model.predict_proba(df)[:, 1][0]
        pred = int(proba >= 0.5)
        st.success(f"Probability of purchase: **{proba:.2%}** — Prediction: **{'Will Purchase' if pred==1 else 'Will Not Purchase'}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.exception(e)

