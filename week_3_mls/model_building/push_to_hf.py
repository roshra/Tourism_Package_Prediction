import os, sys
from huggingface_hub import HfApi, create_repo

HF_TOKEN  = os.getenv("HF_TOKEN", "").strip()
SPACE_ID  = os.getenv("HF_SPACE_ID", "").strip()  # e.g. "roshra/tourism-wellness-app"

if not HF_TOKEN:
    sys.exit("❌ HF_TOKEN not set. In Colab, run:\n%env HF_TOKEN=your_write_token")
if not SPACE_ID or "/" not in SPACE_ID:
    sys.exit("❌ HF_SPACE_ID must look like 'username/space-name'. Example:\n%env HF_SPACE_ID=roshra/tourism-wellness-app")

api = HfApi(token=HF_TOKEN)

# Pick SDK
sdk = "docker" if os.path.exists("week_3_mls/model_building/Dockerfile") else "streamlit"
print(f"ℹ️ Desired Space SDK: {sdk}")

# 1) Ensure the Space exists
try:
    api.repo_info(SPACE_ID, repo_type="space")
    print(f"✅ Space exists: {SPACE_ID}")
except Exception as e_info:
    print(f"ℹ️ Creating Space: {SPACE_ID} (sdk={sdk})")
    try:
        api.create_space(repo_id=SPACE_ID, sdk=sdk, private=False)
        print("✅ Space created via create_space()")
    except Exception as e_cs:
        print("⚠️ create_space failed:", e_cs)
        try:
            create_repo(repo_id=SPACE_ID, repo_type="space", private=False, space_sdk=sdk)
            print("✅ Space created via legacy create_repo()")
        except Exception as e_cr:
            sys.exit(f"❌ Failed to create Space:\n{e_info}\n{e_cs}\n{e_cr}")

# 2) Upload deployment files
files = [
    ("week_3_mls/model_building/app.py", "app.py"),
    ("week_3_mls/model_building/requirements.txt", "requirements.txt"),
    ("week_3_mls/model_building/Dockerfile", "Dockerfile"),
]
if os.path.exists("models/best_pipeline.joblib"):
    files.append(("models/best_pipeline.joblib", "best_pipeline.joblib"))

for local, remote in files:
    if os.path.exists(local):
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=SPACE_ID,
            repo_type="space",
        )
        print(f"📤 Uploaded {remote}")
    else:
        print(f"⚠️ Missing {local}, skipped")

print("✅ Deployment complete:", SPACE_ID)


