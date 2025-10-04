import os, joblib, json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_ID   = os.getenv("HF_MODEL_ID", "roshra/tourism-wellness-xgb")
DATASET_ID = os.getenv("HF_DATASET_ID", "roshra/tourism-wellness-dataset")

# Load splits (prefer local outputs from prep step)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# Identify columns
cat_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
num_cols = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# Candidate models and small grids (kept modest to keep CI fast)
candidates = {
    "dt": (DecisionTreeClassifier(random_state=42),
           {"model__max_depth":[3,5,7]}),
    "rf": (RandomForestClassifier(random_state=42, n_estimators=200),
           {"model__max_depth":[5,10], "model__min_samples_split":[2,5]}),
    "gb": (GradientBoostingClassifier(random_state=42),
           {"model__n_estimators":[150,200], "model__learning_rate":[0.05,0.1]}),
    "xgb": (XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", random_state=42),
            {"model__n_estimators":[150,200], "model__max_depth":[3,4], "model__learning_rate":[0.05,0.1]}),
}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism_pred_prod_taken")

best_name, best_score, best_est = None, -1, None
results = {}

for name, (clf, grid) in candidates.items():
    pipe = Pipeline([("pre", pre), ("model", clf)])
    gs = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1, verbose=0)

    with mlflow.start_run(run_name=name):
        gs.fit(Xtrain, ytrain)
        ypred = gs.best_estimator_.predict(Xtest)
        acc = accuracy_score(ytest, ypred)
        f1  = f1_score(ytest, ypred)

        # Log params & metrics
        for k,v in gs.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        results[name] = {"params": gs.best_params_, "accuracy": acc, "f1": f1}
        if f1 > best_score:
            best_score, best_name, best_est = f1, name, gs.best_estimator_

print("Model Scores:", json.dumps(results, indent=2))
print(f"Best model: {best_name} with F1={best_score:.4f}")

# Save the best model
os.makedirs("models", exist_ok=True)
model_path = "models/best_pipeline.joblib"
import joblib as _joblib
_joblib.dump(best_est, model_path)
print("Saved best model to", model_path)

# Push to HF Model Hub if token provided
if HF_TOKEN:
    try:
        api = HfApi(token=HF_TOKEN)
        try:
            api.repo_info(repo_id=MODEL_ID, repo_type="model")
            print("✅ HF model repo exists:", MODEL_ID)
        except Exception:
            create_repo(repo_id=MODEL_ID, repo_type="model", private=False)
            print("ℹ️ Created HF model repo:", MODEL_ID)

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_pipeline.joblib",
            repo_id=MODEL_ID, repo_type="model"
        )

        # Minimal README model card
        card = f"""# Tourism Wellness Binary Classifier
Best model: {best_name}
Metrics (on holdout): F1={best_score:.4f}
"""
        with open("models/README.md","w") as f:
            f.write(card)
        api.upload_file(
            path_or_fileobj="models/README.md",
            path_in_repo="README.md",
            repo_id=MODEL_ID, repo_type="model"
        )
        print("✅ Uploaded model to HF model hub.")
    except Exception as e:
        print("⚠️ HF model upload failed:", e)
else:
    print("ℹ️ HF_TOKEN not set; skipped model push.")
