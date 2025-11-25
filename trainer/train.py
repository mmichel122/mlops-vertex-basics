import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import storage


# -----------------------------
# Load dataset from GCS or local
# -----------------------------
def load_dataset(dataset_path: str) -> pd.DataFrame:
    print(f"📥 Loading dataset from: {dataset_path}")

    if dataset_path.startswith("gs://"):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------
# Upload local file to GCS using google-cloud-storage
# --------------------------------------------------
def upload_to_gcs(local_path: str, gcs_uri: str):
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"GCS URI must start with gs://, got: {gcs_uri}")

    print(f"⬆ Uploading model to: {gcs_uri}")

    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    object_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_filename(local_path)

    print(f"✅ Model uploaded: {gcs_uri}")


# ---------------------------------------------
# Save model → local then upload to AIP_MODEL_DIR
# ---------------------------------------------
def save_model(model, filename="model.joblib"):
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")

    if not aip_model_dir:
        raise RuntimeError(
            "❌ AIP_MODEL_DIR is not set! "
            "Did you specify a 'Model output directory' in Vertex AI?"
        )

    # Ensure working local filename
    local_path = filename
    joblib.dump(model, local_path)
    print(f"💾 Local model saved as: {local_path}")

    # Build real GCS destination
    # Example AIP_MODEL_DIR:
    #   gs://bucket/vertex-ai/job-output/<job_id>
    gcs_uri = f"{aip_model_dir.rstrip('/')}/{filename}"

    upload_to_gcs(local_path, gcs_uri)


# -----------------
# Main training loop
# -----------------
def main(dataset_path: str, n_estimators: int):
    df = load_dataset(dataset_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"🌲 Training RandomForest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    # Accuracy
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Accuracy: {acc}")

    # Save model
    save_model(clf)
    print("✅ Training job completed successfully.")


# -----------------
# Argument parsing
# -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset CSV on GCS (gs://...) or local.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="RandomForest number of trees.",
    )

    args = parser.parse_args()

    main(args.dataset_path, args.n_estimators)
