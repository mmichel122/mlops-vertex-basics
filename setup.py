from setuptools import setup, find_packages

setup(
    name="training_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
        "gcsfs",
    ],
    python_requires=">=3.10",
)