import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Tourism Wellness Predictor")

# --- Locate model file (root or models/) ---
MODEL_CANDIDATES = ["best_pipeline.joblib", "models/best_pipeline.joblib"]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if MODEL_PATH is None:
    st.error("best_pipeline.joblib not found in Space. Push it to Space root or models/.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.caption(f"Loaded model: `{MODEL_PATH}`")

# ---- Introspect the ColumnTransformer to see what the model expects
try:
    pre: ColumnTransformer = model.named_steps["pre"]
    cats = []
    nums = []
    for name, tr, cols in pre.transformers_:
        if name == "cat" and isinstance(tr, OneHotEncoder):
            cats = list(cols)
        elif name == "num":
            # remainder could be 'drop' or a list; we only list assigned columns
            nums = list(cols) if cols is not None else []
    st.info(f"Model expects categorical columns (OHE): {cats}")
    st.info(f"Model expects numeric columns: {nums}")
except Exception as e:
    st.warning(f"Could not inspect preprocessor: {e}")

# ---- Build the input UI using the full schema shown above
# Use the same lists (cats, nums) to build inputs, so we always send all columns.
# Provide sensible defaults and enforce dtype before prediction.

# Simple default choices for common categoricals:
choices = {
    "TypeofContact": ["Company Invited","Self Enquiry"],
    "Occupation": ["Salaried","Small Business","Large Business","Freelancer","Others"],
    "Gender": ["Male","Female"],
    "MaritalStatus": ["Single","Married","Divorced"],
    "ProductPitched": ["Basic","Standard","Deluxe","Super Deluxe","King"],
    "Designation": ["Executive","Senior Manager","Manager","AVP","VP"],
}

st.subheader("Input")
row = {}

# Categorical inputs:
for c in cats:
    opts = choices.get(c)
    if opts:
        row[c] = st.selectbox(c, opts)
    else:
        row[c] = st.text_input(c, "Unknown")

# Numeric inputs (with safe defaults):
num_defaults = {
    "Age": 35, "CityTier": 2, "DurationOfPitch": 30, "NumberOfTrips": 2,
    "NumberOfFollowups": 2, "PreferredPropertyStar": 3, "NumberOfPersonVisiting": 2,
    "NumberOfChildrenVisiting": 0, "MonthlyIncome": 50000,
    "PitchSatisfactionScore": 3, "Passport": 0, "OwnCar": 0
}
for n in nums:
    # integer-like fields get integer inputs; others number_input
    if n in {"Age","CityTier","NumberOfTrips","NumberOfFollowups","PreferredPropertyStar",
             "NumberOfPersonVisiting","NumberOfChildrenVisiting","PitchSatisfactionScore",
             "Passport","OwnCar"}:
        row[n] = st.number_input(n, value=int(num_defaults.get(n, 0)))
    else:
        row[n] = st.number_input(n, value=float(num_defaults.get(n, 0.0)))

# Build DF with ALL expected columns, in the exact order:
df = pd.DataFrame([{**{c: row.get(c, "Unknown") for c in cats},
                    **{n: row.get(n, num_defaults.get(n, 0)) for n in nums}}])

# Enforce dtypes according to the model’s preprocessor:
for c in cats:
    df[c] = df[c].astype("string")
for n in nums:
    df[n] = pd.to_numeric(df[n], errors="coerce")

st.write("### Row sent to the model")
st.dataframe(df)

if st.button("Predict"):
    try:
        proba = model.predict_proba(df)[:, 1][0]
        pred = int(proba >= 0.5)
        st.success(f"Probability of purchase: **{proba:.2%}** — Prediction: **{'Will Purchase' if pred==1 else 'Will Not Purchase'}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

