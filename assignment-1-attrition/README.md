# Employee Attrition Prediction

## Problem Overview
This project aims to build a machine learning model to predict employee attrition (whether an employee is likely to leave the company) and to identify the key drivers behind attrition. The goal is to help HR teams take proactive retention actions based on data-driven insights.

## Setup Steps
1. Clone the repository or download the assignment folder.
2. Ensure you have Python 3.10+ installed.

3. Install dependencies using either [uv](https://github.com/astral-sh/uv) or pip:
  - With uv (recommended):
    ```bash
    uv pip install -r requirements.txt
    # or, if using pyproject.toml:
    uv pip install
    ```
  - With pip:
    ```bash
    pip install -r requirements.txt
    ```
  (See requirements.txt for the full list of packages.)
4. Launch Jupyter Notebook:
  ```bash
  jupyter notebook assignment-1-attrition.ipynb
  ```

## Instructions to Run the Notebook
- Open `assignment-1-attrition.ipynb` in Jupyter.
- Run all cells sequentially to:
  - Load and explore the dataset
  - Train and evaluate classification models
  - Analyze feature importance and SHAP values
  - Use the sample prediction function for new employee data

## Summary of Key Decisions
- Used both Logistic Regression and Random Forest as baseline models.
- Applied a pipeline for preprocessing (scaling, encoding) and modeling.
- Evaluated models using Accuracy, Precision, Recall, F1, and ROC-AUC due to class imbalance.
- Used feature importance and SHAP for model explainability.
- Provided a functional prediction output for new employee data.

## Expected Outputs
- **Trained Models:** Logistic Regression and Random Forest classifiers.
- **Comparison Table:** Evaluation metrics for both models.
- **Top Attrition Drivers:** Ranked by feature importance and SHAP analysis.
- **Functional Prediction Output:**
  - See the `predict_attrition_probability` function and example usage in the notebook.
  - Example:
    ```python
    sample_employee = pd.DataFrame({
        'age': [29],
        'gender': ['Male'],
        'education': ['Bachelor'],
        'department': ['Sales'],
        'job_role': ['Sales Executive'],
        'years_at_company': [2],
        'promotions': [0],
        'overtime': ['Yes'],
        'performance_rating': [3],
        'monthly_income': [3500]
    })
    prob = predict_attrition_probability(sample_employee)
    print(f"Attrition Probability: {prob:.2f}")
    ```

## Extras (Optional, not fully implemented)
- Hyperparameter tuning
- Advanced imbalance handling (SMOTE, class weights)
- UI/API for predictions

## Notes
- The notebook is self-contained and demonstrates all required steps and outputs as per the assignment statement.
- For further improvements, consider adding hyperparameter tuning and advanced imbalance handling.
