# Data prep: load from HF dataset (preferred) or local, clean, split, upload splits
import os, io, requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_url

HF_TOKEN = os.getenv("HF_TOKEN", "")
DATASET_ID = os.getenv("HF_DATASET_ID", "roshra/tourism-wellness-dataset")
api = HfApi(token=HF_TOKEN)

def load_from_hf():
    try:
        url = hf_hub_url(DATASET_ID, filename="tourism.csv", repo_type="dataset")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print("⚠️ HF load failed, will try local file. Error:", e)
        return None

def load_local():
    p = "week_3_mls/data/tourism.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

# ---- Load dataset ----
df = load_from_hf()
if df is None:
    df = load_local()
if df is None:
    raise FileNotFoundError("tourism.csv not found in HF dataset or local path.")

print("✅ Loaded dataset shape:", df.shape)

# ---- Drop junk/identifiers if present ----
for col in ["Unnamed: 0", "CustomerID"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ---- Ensure target column ----
if "ProdTaken" not in df.columns:
    raise ValueError("Expected 'ProdTaken' target column not found.")

# ---- Encode categorical columns ----
cat_cols = [
    "TypeofContact","Occupation","Gender","MaritalStatus",
    "ProductPitched","Designation"
]
for c in cat_cols:
    if c in df.columns and df[c].dtype == object:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

# ---- Handle missing values ----
df = df.dropna()

# ---- Split into X/y ----
y = df["ProdTaken"].astype(int)
X = df.drop(columns=["ProdTaken"])

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Save locally ----
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("✅ Saved splits locally: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

# ---- Upload back to HF dataset if token provided ----
if HF_TOKEN:
    for fp in ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]:
        api.upload_file(
            path_or_fileobj=fp,
            path_in_repo=fp,
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
    print(f"✅ Uploaded splits to HF dataset: {DATASET_ID}")
else:
    print("ℹ️ HF_TOKEN not set; skipped upload of splits.")

