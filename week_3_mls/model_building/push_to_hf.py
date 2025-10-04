import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN","")
SPACE_ID = os.getenv("HF_SPACE_ID","roshra/tourism-wellness-app")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Configure a write token in env / GitHub Secret.")

api = HfApi(token=HF_TOKEN)

# Ensure Space exists
try:
    api.repo_info(SPACE_ID, repo_type="space")
    print("✅ Space exists:", SPACE_ID)
except Exception:
    print("ℹ️ Creating Space:", SPACE_ID)
    create_repo(SPACE_ID, repo_type="space", private=False, space_sdk="streamlit")

# Upload deployment files
for local, remote in [
    ("week_3_mls/model_building/app.py", "app.py"),
    ("week_3_mls/model_building/requirements.txt", "requirements.txt"),
    ("week_3_mls/model_building/Dockerfile", "Dockerfile"),
    ("models/best_pipeline.joblib", "best_pipeline.joblib"),
]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=SPACE_ID, repo_type="space")
    print("⬆️", remote)

print("✅ Pushed app to HF Space:", SPACE_ID)
