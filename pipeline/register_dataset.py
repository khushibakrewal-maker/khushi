import os
from huggingface_hub import HfApi, hf_hub_download


# Hugging Face dataset repository
HF_DATASET_REPO = "Khushibk/tourism-dataset"

# Original dataset file
DATASET_FILE = "tourism.csv"


def main():

    # Get Hugging Face token from GitHub Secrets
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN is not available.")

    print("Checking Hugging Face dataset repository...")

    # Connect to Hugging Face
    api = HfApi(token=token)

    # Download the dataset from Hugging Face
    local_file = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        token=token
    )

    print("Dataset downloaded successfully:")
    print(local_file)

    # Upload the dataset back to Hugging Face
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=DATASET_FILE,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    print("Dataset registered successfully on Hugging Face.")
    print(f"Repository: {HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
