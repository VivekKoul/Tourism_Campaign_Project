# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score

# for model serialization
import joblib

# huggingface hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Dataset paths (Hugging Face Datasets)
Xtrain_path = "hf://datasets/KoulVivek/tourism_project/Xtrain.csv"
Xtest_path = "hf://datasets/KoulVivek/tourism_project/Xtest.csv"
ytrain_path = "hf://datasets/KoulVivek/tourism_project/ytrain.csv"
ytest_path = "hf://datasets/KoulVivek/tourism_project/ytest.csv"

# Load data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze().astype(int)
ytest = pd.read_csv(ytest_path).squeeze().astype(int)

# Features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Compute class imbalance ratio for XGBoost
neg, pos = ytrain.value_counts()
scale_pos_weight = neg / pos
print("scale_pos_weight =", scale_pos_weight)

# XGBoost classifier
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [75, 100, 150],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.5, 0.6],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# MLflow run
with mlflow.start_run():

    # Grid search
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring=make_scorer(f1_score),
        n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination correctly
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Nested MLflow run for each combination
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Custom threshold
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    # Classification metrics
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    # Save the model locally
    model_path = "tourism_project_model_v2.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as: {model_path}")

    # Upload to Hugging Face Models repo
    HF_MODEL_REPO = "KoulVivek/tourism_project"

    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        print("Model repo already exists.")
    except RepositoryNotFoundError:
        print("Model repo not found. Creating...")
        create_repo(HF_MODEL_REPO, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    print("Model uploaded to Hugging Face Hub successfully.")
