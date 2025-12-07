import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Dataset paths (Hugging Face)
Xtrain_path = "hf://datasets/KoulVivek/tourism_project/Xtrain.csv"
Xtest_path = "hf://datasets/KoulVivek/tourism_project/Xtest.csv"
ytrain_path = "hf://datasets/KoulVivek/tourism_project/ytrain.csv"
ytest_path = "hf://datasets/KoulVivek/tourism_project/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)["ProdTaken"]
ytest = pd.read_csv(ytest_path)["ProdTaken"]

# Categorical features
categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender',
    'MaritalStatus', 'Designation'
]

# Convert categorical columns to string (CatBoost requirement)
for col in categorical_features:
    Xtrain[col] = Xtrain[col].astype(str)
    Xtest[col] = Xtest[col].astype(str)

# Class imbalance handling
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# CatBoost model
model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    class_weights=[1, class_weight],
    verbose=100
)

with mlflow.start_run():

    model.fit(
        Xtrain, ytrain,
        cat_features=categorical_features,
        eval_set=(Xtest, ytest)
    )

    # Predict with threshold = 0.45
    threshold = 0.45

    y_pred_train_proba = model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= threshold).astype(int)

    y_pred_test_proba = model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)

    # Classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log MLflow metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the CatBoost model
    model_path = "tourism_project_catboost_model.cbm"
    model.save_model(model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as {model_path}")

    # Upload to HuggingFace
    repo_id = "KoulVivek/tourism_project_catboost"
    
    try:
        api.repo_info(repo_id, repo_type="model")
        print(f"Model repo '{repo_id}' exists.")
    except RepositoryNotFoundError:
        print(f"Creating new repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model"
    )

    print("Model uploaded to HuggingFace.")
