import os
import pandas as pd

from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download


# Hugging Face dataset repository
HF_DATASET_REPO = "Khushibk/tourism-dataset"

# Dataset file
DATASET_FILE = "tourism.csv"

# Folder where prepared files will be stored
OUTPUT_DIR = "artifacts"


def main():

    # Get Hugging Face token
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN is not available.")

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading tourism dataset from Hugging Face...")

    # Download original dataset
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        token=token
    )

    # Load dataset
    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)

    # --------------------------------------------------
    # Remove unnecessary columns
    # --------------------------------------------------

    columns_to_drop = [
        "Unnamed: 0",
        "CustomerID"
    ]

    existing_columns = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df_clean = df.drop(
        columns=existing_columns
    )

    # --------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------

    before_duplicates = len(df_clean)

    df_clean = df_clean.drop_duplicates()

    after_duplicates = len(df_clean)

    print(
        "Duplicates removed:",
        before_duplicates - after_duplicates
    )

    # --------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------

    categorical_columns = df_clean.select_dtypes(
        include="object"
    ).columns

    for col in categorical_columns:

        df_clean[col] = (
            df_clean[col]
            .str.strip()
        )

    # --------------------------------------------------
    # Fix Gender category
    # --------------------------------------------------

    if "Gender" in df_clean.columns:

        df_clean["Gender"] = df_clean[
            "Gender"
        ].replace(
            {
                "Fe Male": "Female"
            }
        )

    # --------------------------------------------------
    # Check missing values
    # --------------------------------------------------

    print("\nMissing values:")

    print(
        df_clean.isnull().sum()
    )

    # --------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------

    cleaned_path = (
        f"{OUTPUT_DIR}/cleaned_tourism.csv"
    )

    df_clean.to_csv(
        cleaned_path,
        index=False
    )

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    X = df_clean.drop(
        columns=["ProdTaken"]
    )

    y = df_clean["ProdTaken"]

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # --------------------------------------------------
    # Create train dataset
    # --------------------------------------------------

    train_df = X_train.copy()

    train_df["ProdTaken"] = y_train

    # --------------------------------------------------
    # Create test dataset
    # --------------------------------------------------

    test_df = X_test.copy()

    test_df["ProdTaken"] = y_test

    # --------------------------------------------------
    # Save train and test datasets
    # --------------------------------------------------

    train_path = (
        f"{OUTPUT_DIR}/train.csv"
    )

    test_path = (
        f"{OUTPUT_DIR}/test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nData preparation completed.")

    print(
        "Cleaned data:",
        df_clean.shape
    )

    print(
        "Training data:",
        train_df.shape
    )

    print(
        "Testing data:",
        test_df.shape
    )

    print("\nFiles created:")

    print(cleaned_path)

    print(train_path)

    print(test_path)


if __name__ == "__main__":

    main()
