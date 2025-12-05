# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Scaling and Encoding features in the dataset
from sklearn.compose import make_column_transformer # For transforming columns in the dataset
from sklearn.pipeline import make_pipeline # For Pipeline creation
# for model training, tuning, and evaluation
import xgboost as xgb # We will use XGBoost model
from sklearn.model_selection import GridSearchCV # Grid Search for Cross validation
from sklearn.metrics import accuracy_score, classification_report, recall_score # Importing for Metrics of the model
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# for logging metrics using mlflow
import mlflow

#mlflow.set_tracking_uri("http://localhost:5000")
#MLflow will save metrics and params into the GitHub workspace folders.
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Dataset paths (Hugging Face Datasets)
Xtrain_path = "hf://datasets/KoulVivek/tourism_project/Xtrain.csv"
Xtest_path = "hf://datasets/KoulVivek/tourism_project/Xtest.csv"
ytrain_path = "hf://datasets/KoulVivek/tourism_project/ytrain.csv"
ytest_path = "hf://datasets/KoulVivek/tourism_project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

#Target feature
target = 'ProdTaken'

#Converting dataframe to series
ytrain = ytrain[target]
ytest = ytest[target]

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation'
]

# Setting the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Defining hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Starting MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Logging all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Logging each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Logging best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Storing and evaluating the best model
    best_model = grid_search.best_estimator_

    # We will use threshold of 0.45 in our scenario
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    #Setting predicted values either 0 or 1 based on our threshold value
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    #Setting predicted values either 0 or 1 based on our threshold value
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Logging the metrics for the best model
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

    # Saving the model locally
    model_path = "tourism_project_model_v2.joblib"
    joblib.dump(best_model, model_path)

    # Logging the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Uploading to Hugging Face
    repo_id = "KoulVivek/tourism_project"
    repo_type = "model"

    #Checking if the model exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model '{repo_id}' not found. Creating new model...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model '{repo_id}' created.")
    
    #Uploading the model
    api.upload_file(
        path_or_fileobj="tourism_project_model_v2.joblib",
        path_in_repo="tourism_project_model_v2.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )