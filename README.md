# Vertex AI — End-to-End Custom Training & Deployment Pipeline

This project demonstrates a full MLOps workflow on **Google Cloud Vertex AI**, including:

* Dataset storage in GCS
* Creating a Python training package
* Running a Vertex AI Custom Training Job
* Uploading the trained model to Vertex AI Model Registry
* Deploying the model to a real-time prediction endpoint
* Querying the endpoint from a local Python client

This README covers the full lifecycle.

---

# 📁 Project Structure

```
.
├── trainer/
│   ├── __init__.py
│   └── train.py
├── setup.py
├── requirements.txt
├── MANIFEST.in
├── create_package.sh
├── dist/
│   └── training_package-0.1.tar.gz
└── predict_client.py
```

---

# 1️⃣ Dataset Setup

Your dataset is stored in Google Cloud Storage:

```
gs://mlops-vertex-training/datasets/iris.csv
```

This path is passed to the training job using an argument:

```
--dataset-path=gs://mlops-vertex-training/datasets/iris.csv
```

---

# 2️⃣ Python Training Package

## `setup.py`

Pinned to match the Vertex runtime:

```python
from setuptools import setup, find_packages

setup(
    name="training_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas==2.0.3",
        "scikit-learn==1.0.2",
        "joblib==1.1.0",
        "gcsfs==2023.6.0",
    ],
    python_requires=">=3.8,<3.11",
)
```

---

# 3️⃣ Trainer Code (`trainer/train.py`)

Supports:

* Loading CSV from GCS
* Arguments from Vertex AI
* Saving the trained model locally
* Uploading it to GCS **or** using `AIP_MODEL_DIR` inside Vertex

Key features:

* Uses `joblib`
* Traces training progress with clean logs
* Fully compatible with Scikit-learn prebuilt containers

---

# 4️⃣ Packaging the Code

```
python setup.py sdist
```

This produces:

```
dist/training_package-0.1.tar.gz
```

---

# 5️⃣ Upload Training Package to GCS

```
gsutil cp dist/training_package-0.1.tar.gz \
  gs://mlops-vertex-training/training/
```

---

# 6️⃣ Vertex AI Custom Training Job

Example job configuration:

* **Container:** `us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest`
* **Python Module:** `trainer.train`
* **Package URI:**
  `gs://mlops-vertex-training/training/training_package-0.1.tar.gz`
* **Arguments:**

  ```
  --dataset-path=gs://mlops-vertex-training/datasets/iris.csv
  --n-estimators=150
  ```

The job:

✔ Installs your package
✔ Loads dataset from GCS
✔ Trains classifier
✔ Saves `model.joblib` locally
✔ Uploads final model to:

```
gs://mlops-vertex-training/models/model/model.joblib
```

---

# 7️⃣ Import Model into Vertex AI Model Registry

Go to:

**Vertex AI → Models → Import**

Specify:

```
gs://mlops-vertex-training/models/model/
```

Vertex loads:

* model.joblib
* Creates a model version
* Prepares it for deployment

---

# 8️⃣ Deploy Model to Endpoint

Options used:

* **Machine:** `e2-standard-2`
* **Min nodes:** `0`
* **Max nodes:** `1`
* **Server type:** *Auto-selected by Vertex*
* **Explainability:** Disabled
* **Encryption:** Google-managed

Resulting endpoint (example):

```
projects/991706256484/locations/europe-west4/endpoints/6470364245694349312
```

Deployment produces a live HTTPS endpoint.

---

# 9️⃣ Query the Endpoint Locally

## Install dependencies

```
pip install google-cloud-aiplatform
```

## Run prediction client

```
python predict_client.py \
  --sepal-length 6.0 \
  --sepal-width 2.2 \
  --petal-length 4.0 \
  --petal-width 1.0
```

---

# 🔟 Example Client Output

```
Using endpoint: projects/.../endpoints/6470364245694349312
Prediction(predictions=['versicolor'], ...)
```

---

# 🧪 Example Inputs for Testing

### Versicolor

```
[6.0, 2.2, 4.0, 1.0]
```

### Virginica

```
[7.1, 3.0, 5.9, 2.1]
```

### Setosa

```
[5.1, 3.5, 1.4, 0.2]
```

---

# 🧹 Cleanup Resources

### Delete model versions

```
gcloud ai models versions delete MODEL_VERSION_ID \
  --model=MODEL_ID --region=europe-west4
```

### Delete endpoint

If stuck in FAILED state, GCP auto-cleans in ~7 days.

---

# ✅ Summary

This pipeline demonstrates:

✔ Vertex AI dataset management
✔ Fully custom training in prebuilt containers
✔ Packaging Python code for distributed execution
✔ GCS → Train → Model Registry → Endpoint workflow
✔ Secure scalable real-time prediction

You now have a **complete production-grade MLOps workflow on Vertex AI**.

---

If you want, I can generate:

🔹 Architecture diagram
🔹 Terraform to automate everything
🔹 A CI/CD pipeline (GitHub Actions / Cloud Build)
🔹 A FastAPI prediction gateway
🔹 A Streamlit front-end for the model
