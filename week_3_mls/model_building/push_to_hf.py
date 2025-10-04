import os, sys
from huggingface_hub import HfApi, create_repo

HF_TOKEN  = os.getenv("HF_TOKEN", "")
SPACE_ID  = os.getenv("HF_SPACE_ID", "roshra/tourism-wellness-app").strip()

if not HF_TOKEN:
    sys.exit("❌ HF_TOKEN not set. Configure as a GitHub secret.")
if not SPACE_ID or "/" not in SPACE_ID:
    sys.exit("❌ HF_SPACE_ID must look like 'username/space-name'.")

api = HfApi(token=HF_TOKEN)

# Ensure Space exists (no space_sdk!)
try:
    api.repo_info(SPACE_ID, repo_type="space")
    print(f"✅ Space exists: {SPACE_ID}")
except Exception:
    print(f"ℹ️ Creating Space: {SPACE_ID}")
    create_repo(SPACE_ID, repo_type="space", private=False)
    print("✅ Space created")

# Upload files
for local, remote in [
    ("week_3_mls/model_building/app.py", "app.py"),
    ("week_3_mls/model_building/requirements.txt", "requirements.txt"),
    ("week_3_mls/model_building/Dockerfile", "Dockerfile"),
    ("models/best_pipeline.joblib", "best_pipeline.joblib"),
]:
    if os.path.exists(local):
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=SPACE_ID, repo_type="space")
        print("⬆️", remote)
    else:
        print("⚠️ Missing:", local)

print("✅ Deployment complete:", SPACE_ID)
