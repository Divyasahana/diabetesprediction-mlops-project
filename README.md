# Diabetes Prediction — End-to-End MLOps Project

This project demonstrates a complete **Machine Learning Operations (MLOps) pipeline** for predicting diabetes using the **Pima Indians Diabetes Dataset**. The system follows the full machine learning lifecycle including **data preprocessing, model training, experiment tracking, testing, CI/CD automation, API deployment, containerization, and monitoring**.

The objective is to build a **reproducible and production-ready machine learning system** capable of predicting whether a patient has diabetes based on medical attributes.

---

# 1. Project Overview

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help healthcare professionals take preventive action and reduce severe complications.

This project builds a **machine learning model that predicts diabetes risk using patient health data** and deploys it through a **production-style MLOps pipeline**.

The system demonstrates how machine learning models move from **development to deployment and monitoring**.

## Key Features

- Data preprocessing pipeline  
- Machine learning model training  
- Experiment tracking with MLflow  
- Automated testing with Pytest  
- Continuous Integration using GitHub Actions  
- FastAPI deployment for prediction  
- Docker containerization  
- API monitoring with Prometheus metrics  

The final result is a **deployable ML service** that provides real-time diabetes predictions through an API.

---

# 2. Problem Definition & Data

## Problem Definition

The objective of this project is to **predict whether a patient has diabetes** based on medical diagnostic features.

This is formulated as a **binary classification problem**:
0 → Patient does not have diabetes
1 → Patient has diabetes


The trained machine learning model learns patterns from historical patient data and predicts diabetes risk for new inputs.

---

## Dataset

Dataset used:

**Pima Indians Diabetes Dataset**

The dataset contains medical measurements collected from female patients.

---

## Features
The dataset contains the following input features:

| Feature | Description |
|-------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Serum insulin level |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes probability |
| Age | Age of the patient |

### Target Variable
Outcome

- 0 → No Diabetes
- 1 → Diabetes

---

## Data Processing

Data preprocessing steps include:

- Dataset loading
- Handling missing values
- Feature preparation
- Splitting dataset into training and testing sets

The preprocessing logic is implemented in:
src/preprocess.py

---

# 3. System Architecture

The system follows a **modular MLOps architecture** consisting of multiple components responsible for different stages of the ML lifecycle.

## Architecture Components

**Data Processing Layer**  
Handles dataset loading and preprocessing.

**Model Training Layer**  
Trains a machine learning model using Scikit-learn.

**Experiment Tracking**  
MLflow records parameters, metrics, and artifacts.

**Model Storage**
model/model.pkl


**Prediction API**  
FastAPI exposes the trained model through REST endpoints.

**Containerization**  
Docker packages the application for reproducible deployment.

**Monitoring**  
Prometheus metrics track API performance.

---

## System Architecture Diagram
            +----------------------+
            |   Pima Diabetes Data |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |   Data Preprocessing |
            |   (preprocess.py)    |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |    Model Training    |
            |      (train.py)      |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |  MLflow Experiment   |
            |      Tracking        |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |   Trained Model      |
            |     model.pkl        |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |      FastAPI API     |
            |       /predict       |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |     Docker Image     |
            |  Containerized App   |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Monitoring Metrics   |
            |   /metrics endpoint  |
            |      Prometheus      |
            +----------------------+


---

# 4. MLOps Practices

This project follows several **industry-standard MLOps practices** to ensure reproducibility, reliability, and maintainability.

---

## Version Control

Git and GitHub were used for version control and collaborative development throughout the project.

Feature branches were created for different components of the system to allow team members to work independently without affecting the main codebase.

Example feature branches used in the project include:

- feature/model-improvement
- feature/api-improvement
- feature/docker-setup
- feature/monitoring
- feature/documentation

After completing their tasks, team members created **Pull Requests** to merge their branches into the main branch. The code was reviewed and merged to maintain code quality and project stability.

---

## Experiment Tracking

MLflow is used to track machine learning experiments.

It records:

- Model parameters
- Evaluation metrics
- Experiment runs
- Model artifacts

Start MLflow:

```bash
mlflow ui 
```
Open:
http://127.0.0.1:5000

---

# Automated Testing

Unit tests are implemented using Pytest to ensure system reliability.

## Test files:
tests/test_data_loader.py\
tests/test_preprocess.py

## Run tests:
pytest

## Continuous Integration
A GitHub Actions pipeline automatically performs:

- Code formatting checks
- Linting
- Unit test execution

Configuration file:
.github/workflows/ci.yml

# Model Deployment

The trained model is deployed using FastAPI.

## API file:  
src/api.py

## Run API Locally:
``` bash
uvicorn src.api:app --reload
```

Open API documentation:
http://127.0.0.1:8000/docs

Endpoints:

-   GET / → Health check
-   POST /predict → Returns prediction

---

# Docker Containerization

Docker packages the entire application into a portable container.

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

---

# 5. Monitoring & Reliability

Monitoring is implemented using Prometheus FastAPI Instrumentator.

Metrics endpoint:
http://localhost:8000/metrics

Metrics include:

- API request count
- Request latency
- Endpoint performance

Monitoring helps detect:
- Slow response times
- High traffic load
- API errors

This improves the reliability and maintainability of the deployed ML system.

---

# 6. Team Collaboration

The project was developed collaboratively using **GitHub feature branches and pull requests**.

## Team Responsibilities

| Team Member | Responsibility |
|-------------|---------------|
| Member 1 | Model improvement and evaluation |
| Member 2 | FastAPI API development |
| Member 3 | Docker containerization |
| Member 4 | Monitoring and metrics integration |
| Member 5 | Documentation and report preparation |

All contributions were merged through **Pull Requests and code reviews** to ensure code quality and maintain project consistency.

---

# 7. Limitations & Future Work
## Limitations

Although the project demonstrates a full MLOps pipeline, several limitations remain:
- Dataset size is limited
- Model performance can be improved
- Monitoring does not include alerting mechanisms
- No automated model retraining pipeline

## Future Improvements

Future enhancements could include:

- Larger healthcare datasets
- Automated model retraining pipelines
- Advanced monitoring with alert systems
- Cloud deployment (AWS / GCP / Azure)
- Secure API authentication
- CI/CD deployment pipeline

These improvements would make the system more suitable for **real-world healthcare deployment**.

---

# How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Divyasahana/diabetesprediction-mlops-project.git
```
### 2. Move to the project directory
``` bash
cd diabetesprediction-mlops-project
```

### 3. Create a virtual environment
``` bash
uv venv --python 3.11
```

### 4. Activate the environment (Windows)
``` bash
.venv\Scripts\activate
```

### 5. Install dependencies
``` bash
uv sync
```

### 6. Train the model
``` bash
python src/train.py
```

### 7. Start MLflow
``` bash
mlflow ui
```

Open in browser:
http://127.0.0.1:5000

### 8. Start the API
``` bash
uvicorn src.api:app --reload
```

Open API documentation:
http://127.0.0.1:8000/docs

---

# Tools & Technologies

The project uses the following tools and technologies:

- Python 3.11
- Scikit-learn
- MLflow
- FastAPI
- Pytest
- GitHub Actions
- Docker
- Prometheus Monitoring
- Black (code formatter)
- Ruff (linter)
- UV Package Manager

---

# License

This project is developed for **academic purposes** as part of an MLOps coursework project.


