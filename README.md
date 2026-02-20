# Diabetes Prediction – MLOps Project

## Team Member Roles

| Team Member |
|------------|
| Divya JAYAPRAKASH |
| Jayasri DHANAPAL |
| Reshma KARTHIKEYAN NAIR |
| Gurpreetkaur Jaykumar MODI |
| Vidya Sandeep NAKADE |


## Project Description
This project aims to build an end-to-end MLOps pipeline to predict whether a patient has diabetes using medical diagnostic measurements.
The focus is on reproducibility, clean code structure, and MLOps best practices rather than model complexity.

## Machine Learning Task
- **Problem type:** Binary classification
- **Objective:** Predict diabetes outcome (0 = No diabetes, 1 = Diabetes)
- **Model (baseline):** Logistic Regression

## Dataset
- **Source:** https://github.com/plotly/datasets/blob/master/diabetes.csv
- **Features:** Medical attributes such as glucose level, BMI, age, etc.
- **Target:** `Outcome`


## Project Structure
DiabetesPrediction-mlops-project/
│
├── data/
│ └── diabetes.csv
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ └── train.py
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore

## Dataset Description

This project uses the Pima Indians Diabetes Dataset for binary classification.

| Column Name | Description |
|------------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Target variable (0 = Non-diabetic, 1 = Diabetic) |

## How to Run the Project Locally

### 1. Clone the repository
```bash`
git clone https://github.com/Divyasahana/DiabetesPrediction-mlops-project.git
cd DiabetesPrediction-mlops-project

### 2. Create and activate virtual environment
```bash`
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

### 3. Install dependencies
pip install .

### 4. Run the application
python src/train.py


---


## Project Workflow

1. Load dataset from `data/diabetes.csv`
2. Preprocess and clean data (handle missing values, scale features)
3. Train machine learning model (Logistic Regression baseline)
4. Evaluate model performance (accuracy, confusion matrix)
5. Save trained model for deployment

# Experiment Tracking

This project uses **MLflow** to track machine learning experiments, including:

- Model parameters
- Evaluation metrics
- Trained model artifacts

### Experiment Details

- **Experiment Name:** `diabetes-baseline`
- **Tracked Metric:** Accuracy
- **Artifacts Logged:** Trained model file

### Run locally

```bash
python src/train.py
mlflow ui

###Quick Verification Commands:
```bash
# Clone and setup
git clone https://github.com/Divyasahana/diabetesprediction-mlops-project
cd diabetesprediction-mlops-project
uv sync

# Verify pre-commit hooks
pre-commit run --all-files

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run training pipeline
python -m src.train

# View MLflow experiments
mlflow ui

### Instruction
 Run training script before running API.
git checkout feature/api-checkpoint3
