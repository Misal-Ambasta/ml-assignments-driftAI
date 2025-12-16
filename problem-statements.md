# Assignment 1 - Employee Attrition Prediction

>Your Company is expanding rapidly, but HR has noticed a rising trend of employees leaving unexpectedly. The leadership team wants to shift from reactive decision-making to a predictive model that can identify employees who may be at risk of leaving the company. The goal is to better understand the underlying reasons behind attrition and take early action to improve retention.

You are assigned to build a machine-learning model that analyzes employee data and predicts the likelihood of attrition, along with insights on the factors contributing to it.

## Requirements
- Load and prepare the `attrition.csv` dataset
- Perform essential data cleaning and basic EDA
- Encode features and prepare data for modeling
- Train at least two classification models
- Evaluate using Accuracy, Precision, Recall, F1, ROC-AUC
- Identify top factors driving attrition

### Extras
- Hyperparameter tuning (improves model performance)
- SHAP / feature-importance visuals (explains model behavior)
- Optional prediction UI/API (demo for real usage)
- Advanced imbalance methods (SMOTE/class weights to improve fairness)

## Data Structure
**1. Personal & Background**
	- `age`: Numerical value representing employee age
	- `gender`: Categorical (Male/Female/Other)
	- `education`: Categorical or ordinal (e.g., Bachelor, Master, PhD)

**2. Job & Work Environment**
	- `department`: Employee’s department
	- `job_role`: Specific role title
	- `years_at_company`: Number of years the employee has spent at DriftAl
	- `promotions`: Count of promotions received
	- `overtime`: Yes/No indicator
	- `performance_rating`: Numeric or categorical performance score
	- `monthly_income`: Numeric salary amount

**3. Target Column**
	- `attrition`: 0 = stays, 1 = leaves

## Expected Output
- A fully trained machine-learning model capable of predicting employee attrition
- A comparison table showing results of all trained models
- A ranked list of top attrition drivers with brief interpretation
- A functional prediction output where, for any given employee, the model should return the probability of that employee leaving the company (attrition probability)

## Documentation Guidelines
- Setup steps
- Instructions to run the notebook
- Summary of key decisions



# Assignment 2 - ML Pipeline Debugging & Data Leakage Detection

>Your company's Analytics team discovered that a model built internally shows extremely high accuracy during testing but fails badly in real scenarios. This suggests the presence of data leakage or incorrect pipeline steps. You have been assigned to review the faulty notebook, identify all issues, and fix the ML workflow.

The objective is to demonstrate the difference between a leaking model and a correctly built model, helping the company avoid such pitfalls in future projects.

## Basic Requirements
- Review the provided notebook and find mistakes
- Identify and explain all forms of data leakage
- Correct preprocessing, splitting, and scaling
- Retrain a clean pipeline
- Compare before vs after results with explanation

### Extras
- sklearn Pipeline (standardized, clean, error-proof workflow)
- Leakage-detection utility (quick checks to detect accidental leakage)
- Model explainability visuals (interpret differences before and after fixing)

## Data Structure
**Features**
	- Numerical fields (age, salary, counts, ratings, etc.)
	- Categorical fields (department, category labels, status)

**Possible Leakage Columns**
	- Fields that directly depend on the target
	- Fields created after the event/outcome
	- Pre-aggregated indicators

**Target Column**
	- `target` (classification or regression depending on notebook)

## Expected Output
- A corrected ML pipeline free from data leakage
- A well-documented list of all mistakes found in the original version
- A performance comparison showing realistic vs inflated accuracy
- A short explanation describing why each correction was necessary

## Documentation Guidelines
- Steps to run the fixed pipeline
- Summary of all corrections
- Short debugging report



# Assignment 3 - Productivity Feature Engineering & Optimization

>Your Operations team wants to build a smarter system that predicts employee productivity. Their current metrics show raw performance numbers, but they believe deeper relationships exist between workload, experience, team size, and project outcomes.

Your task is to create meaningful new features, build a baseline regression model, and then optimize it to achieve significantly better performance. This work will help your operations team understand what really drives productivity across teams.

## Basic Requirements
- Load and clean `employee_productivity.csv`
- Build a baseline regression model
- Engineer multiple new features
- Apply scaling/feature selection where appropriate
- Build an optimized model and compare performance with baseline

### Extras
- Clustering-based features (group employees by similarity to create behavioural segments)
- PCA (Principal Component Analysis) - reduce dimensionality for stability or visualization
- Feature-importance dashboard (clear visual of what affects productivity)

## Data Structure
**1. Work Profile**
	- `years_experience` (numeric)
	- `hours_week` (numeric)
	- `projects_completed` (numeric)

**2. Team Dynamics**
	- `team_size`
	- `team_previous_performance` (optional depending on dataset)

**3. Quality Metrics**
	- `quality_score` (numeric)
	- `error_rate` (optional)

**4. Target Column**
	- `productivity_score` (numeric)

## Expected Output
- A baseline model showing initial predictive power
- An optimized model demonstrating accuracy improvement
- A list of engineered features with one-line explanation for each
- A clear before–after comparison showing how engineering and tuning improved results

## Documentation Guidelines
- Installation instructions
- How to run baseline and optimized models
- Summary of engineered features


