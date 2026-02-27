# Diabetes Prediction -- End-to-End MLOps Project

This project demonstrates a complete MLOps workflow for predicting
diabetes using machine learning.\
It includes model development, experiment tracking, testing, CI/CD
automation, API deployment, and Docker containerization.

------------------------------------------------------------------------

# Dataset Description

The dataset used is the **Pima Indians Diabetes Dataset**.

### Features:

1.  Pregnancies -- Number of times pregnant
2.  Glucose -- Plasma glucose concentration
3.  BloodPressure -- Diastolic blood pressure (mm Hg)
4.  SkinThickness -- Triceps skin fold thickness (mm)
5.  Insulin -- 2-Hour serum insulin (mu U/ml)
6.  BMI -- Body mass index
7.  DiabetesPedigreeFunction -- Diabetes pedigree function
8.  Age -- Age in years

### Target Variable:

-   Outcome (0 = Non-diabetic, 1 = Diabetic)

------------------------------------------------------------------------

#  Checkpoint 1 -- Project Setup & Model Development

## Tasks Completed:

-   Created structured project layout
-   Set up virtual environment using UV
-   Implemented data preprocessing pipeline
-   Trained machine learning model using Scikit-learn
-   Saved trained model as `model.pkl`
-   Verified training script execution

------------------------------------------------------------------------

#  Checkpoint 2 -- MLflow Integration

## Tasks Completed:

-   Integrated MLflow for experiment tracking
-   Logged model parameters and metrics
-   Stored trained model as artifact
-   Managed experiment runs
-   Launched MLflow UI for visualization

### Start MLflow UI:

``` bash
mlflow ui
```

Open browser:

http://127.0.0.1:5000

------------------------------------------------------------------------

#  Checkpoint 3 -- CI/CD, Testing & Automation

## Tasks Completed:

-   Implemented unit tests using pytest
-   Configured test coverage reporting
-   Added pre-commit hooks
-   Integrated Ruff (linting) and Black (formatting)
-   Created GitHub Actions CI pipeline

### Run Tests:

``` bash
pytest
```

### Run with Coverage:

``` bash
pytest --cov=src --cov-report=term
```

------------------------------------------------------------------------

#  FastAPI Model Deployment

The trained model is deployed using FastAPI.

## Run API Locally:

``` bash
uvicorn src.api:app --reload
```

Open:

http://127.0.0.1:8000/docs

Endpoints:

-   GET / → Health check
-   POST /predict → Returns prediction

------------------------------------------------------------------------

#  Docker Setup

The application is containerized using Docker for portability.

## Build Docker Image:

``` bash
docker build -t diabetes-mlops-app .
```

## Run Docker Container:

``` bash
docker run -p 8000:8000 diabetes-mlops-app
```

Application will be available at:

http://localhost:8000

------------------------------------------------------------------------

#  How to Run the Project (Complete Steps)

### 1. Create Virtual Environment

``` bash
uv venv --python 3.11
```

### 2. Activate Environment

Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install Dependencies

``` bash
uv sync
```

### 4. Train Model

``` bash
python src/train.py
```

### 5. Start MLflow

``` bash
mlflow ui
```

### 6. Run API

``` bash
uvicorn src.api:app --reload
```

------------------------------------------------------------------------

# 🛠 Tools & Technologies Used

-   Python 3.11
-   Scikit-learn
-   MLflow
-   Pytest
-   GitHub Actions
-   FastAPI
-   Uvicorn
-   Docker
-   Ruff
-   Black
-   UV Package Manager

------------------------------------------------------------------------

# Team Contributions

-   Divya JAYAPRAKASH -- Model Development
-   Jayasri DHANAPAL -- MLflow Integration
-   Reshma Karthikeyan NAIR -- CI/CD Setup
-   Gurpreetkaur Jaykumar MODI -- Testing & Coverage
-   Vidya Sandeep NAKADE -- Documentation, API & Docker

------------------------------------------------------------------------

# Project Highlights

✔ End-to-end ML pipeline\
✔ Experiment tracking\
✔ Automated testing\
✔ Continuous Integration\
✔ Model deployment as REST API\
✔ Containerized deployment\
✔ Real-world MLOps workflow

------------------------------------------------------------------------

© 2026 Diabetes Prediction MLOps Project
