from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
DATASET_ID = os.getenv("HF_DATASET_ID", "roshra/tourism-wellness-dataset")

api = HfApi(token=HF_TOKEN)

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=DATASET_ID, repo_type="dataset")
    print(f"✅ Dataset repo exists: {DATASET_ID}")
except RepositoryNotFoundError:
    print(f"ℹ️ Creating dataset repo: {DATASET_ID}")
    create_repo(DATASET_ID, repo_type="dataset", private=False)

# Upload all files in data folder
api.upload_folder(
    folder_path="week_3_mls/data",
    repo_id=DATASET_ID,
    repo_type="dataset"
)
print("✅ Uploaded dataset files to HF dataset repo.")
