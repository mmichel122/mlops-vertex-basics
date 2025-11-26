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
)