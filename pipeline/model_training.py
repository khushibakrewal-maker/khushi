import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from huggingface_hub import HfApi


# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_REPO = "Khushibk/tourism-product-prediction"

OUTPUT_DIR = "artifacts"

MODEL_FILE = f"{OUTPUT_DIR}/tourism_best_model.pkl"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Hugging Face authentication
    # --------------------------------------------------------

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN is not available."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load prepared train and test data
    # --------------------------------------------------------

    train_path = f"{OUTPUT_DIR}/train.csv"
    test_path = f"{OUTPUT_DIR}/test.csv"

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    print(
        "Training data shape:",
        train_df.shape
    )

    print(
        "Testing data shape:",
        test_df.shape
    )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=["ProdTaken"]
    )

    y_train = train_df["ProdTaken"]

    X_test = test_df.drop(
        columns=["ProdTaken"]
    )

    y_test = test_df["ProdTaken"]

    # --------------------------------------------------------
    # Identify numerical and categorical columns
    # --------------------------------------------------------

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(
        "Numerical features:",
        numeric_features
    )

    print(
        "Categorical features:",
        categorical_features
    )

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Combined preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Bagging pipeline
    # --------------------------------------------------------

    bag_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                BaggingClassifier(
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------

    bag_params = {

        "model__n_estimators": [
            50,
            100,
            200
        ],

        "model__max_samples": [
            0.7,
            1.0
        ],

        "model__max_features": [
            0.7,
            1.0
        ]
    }

    # --------------------------------------------------------
    # Stratified cross validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # --------------------------------------------------------
    # Grid Search
    # --------------------------------------------------------

    grid = GridSearchCV(
        estimator=bag_pipeline,
        param_grid=bag_params,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print(
        "\nStarting Bagging hyperparameter tuning..."
    )

    grid.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = grid.best_estimator_

    print(
        "\nBest Parameters:"
    )

    print(
        grid.best_params_
    )

    print(
        "\nBest Cross Validation F1:"
    )

    print(
        grid.best_score_
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = best_model.predict(
        X_test
    )

    y_prob = best_model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(
        "Accuracy :",
        accuracy
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall   :",
        recall
    )

    print(
        "F1 Score :",
        f1
    )

    print(
        "ROC-AUC  :",
        roc_auc
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "Tourism Product Prediction"
    )

    with mlflow.start_run(
        run_name="Bagging_Final_Model"
    ):

        # Log tuned parameters
        for parameter, value in grid.best_params_.items():

            mlflow.log_param(
                parameter,
                value
            )

        # Log CV score
        mlflow.log_metric(
            "cv_f1",
            grid.best_score_
        )

        # Log test metrics
        mlflow.log_metric(
            "test_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "test_precision",
            precision
        )

        mlflow.log_metric(
            "test_recall",
            recall
        )

        mlflow.log_metric(
            "test_f1",
            f1
        )

        mlflow.log_metric(
            "test_roc_auc",
            roc_auc
        )

        print(
            "\nMLflow tracking completed."
        )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Upload model to Hugging Face
    # --------------------------------------------------------

    api = HfApi(
        token=token
    )

    api.upload_file(
        path_or_fileobj=MODEL_FILE,
        path_in_repo="tourism_best_model.pkl",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

    print(
        "\nFinal model uploaded successfully."
    )

    print(
        "Hugging Face model repository:",
        HF_MODEL_REPO
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
