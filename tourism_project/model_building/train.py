# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, recall_score # Importing for Metrics of the model

# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


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


# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch',
    'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched','MaritalStatus','Designation'
]
# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Setting the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define base XGBoost Classifier for classification task
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=class_weight)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
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

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Predictions (These predictions are now the thresholded ones for classification report)
    # y_pred_train = best_model.predict(Xtrain) # Removed as we already have thresholded predictions
    # y_pred_test = best_model.predict(Xtest) # Removed as we already have thresholded predictions

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
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

# # Preprocessor
# preprocessor = make_column_transformer(
#     (StandardScaler(), numeric_features),
#     (OneHotEncoder(handle_unknown='ignore'), categorical_features)
# )

# # Define base XGBoost Regressor
# xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# # Hyperparameter grid
# param_grid = {
#     'xgbregressor__n_estimators': [50, 100, 150],
#     'xgbregressor__max_depth': [3, 5, 7],
#     'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
#     'xgbregressor__subsample': [0.7, 0.8, 1.0],
#     'xgbregressor__colsample_bytree': [0.7, 0.8, 1.0],
#     'xgbregressor__reg_lambda': [0.1, 1, 10]
# }

# # Pipeline
# model_pipeline = make_pipeline(preprocessor, xgb_model)

# with mlflow.start_run():
#     # Grid Search
#     grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
#     grid_search.fit(Xtrain, ytrain)

#     # Log parameter sets
#     results = grid_search.cv_results_
#     for i in range(len(results['params'])):
#         param_set = results['params'][i]
#         mean_score = results['mean_test_score'][i]

#         with mlflow.start_run(nested=True):
#             mlflow.log_params(param_set)
#             mlflow.log_metric("mean_neg_mse", mean_score)

#     # Best model
#     mlflow.log_params(grid_search.best_params_)
#     best_model = grid_search.best_estimator_

#     # Predictions
#     y_pred_train = best_model.predict(Xtrain)
#     y_pred_test = best_model.predict(Xtest)

#     # Metrics
#     train_rmse = mean_squared_error(ytrain, y_pred_train, squared=False)
#     test_rmse = mean_squared_error(ytest, y_pred_test, squared=False)

#     train_mae = mean_absolute_error(ytrain, y_pred_train)
#     test_mae = mean_absolute_error(ytest, y_pred_test)

#     train_r2 = r2_score(ytrain, y_pred_train)
#     test_r2 = r2_score(ytest, y_pred_test)

#     # Log metrics
#     mlflow.log_metrics({
#         "train_RMSE": train_rmse,
#         "test_RMSE": test_rmse,
#         "train_MAE": train_mae,
#         "test_MAE": test_mae,
#         "train_R2": train_r2,
#         "test_R2": test_r2
#     })

    # Save the model locally
    model_path = "tourism_project_model_v2.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "KoulVivek/tourism_project"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="tourism_project_model_v2.joblib",
        path_in_repo="tourism_project_model_v2.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
