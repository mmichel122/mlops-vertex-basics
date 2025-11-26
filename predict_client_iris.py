#!/usr/bin/env python3
"""
predict_client.py
------------------
Local client to send prediction requests to a deployed Vertex AI endpoint.

Features:
- Auto-discovers endpoint by display name.
- Validates user input.
- Pretty-prints predictions.
"""

from google.cloud import aiplatform
import argparse
import sys

PROJECT_ID = "mlops-vertex-playground"
LOCATION = "europe-west4"
ENDPOINT_DISPLAY_NAME = "iris-classification-endpoint"


def get_endpoint_by_name(display_name: str):
    """Return the Vertex AI endpoint object for a given display name."""
    print(f"\n🔍 Looking for endpoint named: {display_name}")

    endpoints = aiplatform.Endpoint.list()

    for ep in endpoints:
        if ep.display_name == display_name:
            print(f"✅ Found endpoint: {ep.resource_name}")
            return ep

    print(f"❌ ERROR: No endpoint found with name '{display_name}'")
    sys.exit(1)


def run_prediction(endpoint, features):
    """Send prediction request to Vertex AI."""

    print("\n📤 Sending prediction request...")
    print(f"Input features: {features}")

    response = endpoint.predict(instances=[features])

    print("\n📥 Prediction response:")
    print("----------------------------")
    print(f"Prediction: {response.predictions}")
    print(f"Model version: {response.model_version_id}")
    print(f"Deployed model ID: {response.deployed_model_id}")
    print("----------------------------")

    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using Vertex AI endpoint")

    parser.add_argument("--sepal-length", type=float, default=5.1)
    parser.add_argument("--sepal-width", type=float, default=3.5)
    parser.add_argument("--petal-length", type=float, default=1.4)
    parser.add_argument("--petal-width", type=float, default=0.2)

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert CLI args into a feature list
    features = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    ]

    # Init Vertex AI client
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Auto-select endpoint
    endpoint = get_endpoint_by_name(ENDPOINT_DISPLAY_NAME)

    # Run prediction
    run_prediction(endpoint, features)


if __name__ == "__main__":
    main()
