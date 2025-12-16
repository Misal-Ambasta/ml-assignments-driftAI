# Assignment 2 - ML Pipeline Debugging & Data Leakage Detection

## 📋 Overview

This assignment demonstrates the identification and correction of data leakage issues in a machine learning pipeline. The goal is to show the difference between a broken model with inflated performance metrics and a properly implemented model with realistic results.

---

## 🎯 Objectives

1. **Identify all issues** in the broken notebook (`debug_broken_notebook.ipynb`)
2. **Explain data leakage** and its impact on model performance
3. **Implement a corrected pipeline** free from leakage
4. **Compare performance** between broken and corrected models
5. **Demonstrate best practices** for ML pipeline development

---

## 📁 Files in This Directory

- **`debug_broken_notebook.ipynb`** - Intentionally broken notebook with data leakage issues (DO NOT USE)
- **`assignment-2-debugging.ipynb`** - Corrected implementation with proper ML pipeline ✅
- **`README.md`** - This documentation file

---

## 🔴 Issues Found in Broken Notebook

### Critical Issues:

1. **Target Leakage (Line 67)**
   - **Issue**: Created `attrition_copy` feature directly from target variable
   - **Impact**: Model has direct access to the answer → ~100% accuracy
   - **Why it's wrong**: This feature will not exist in production data

2. **Scaling Before Split (Lines 99-106)**
   - **Issue**: Fitted `StandardScaler` on entire dataset before train/test split
   - **Impact**: Test set statistics leaked into training data
   - **Why it's wrong**: Model sees information from "future" data it shouldn't have access to

3. **Suspicious Feature: `target_leakage_feature`**
   - **Issue**: Used without checking its origin or correlation with target
   - **Impact**: Likely derived from target, causing hidden leakage
   - **Why it's wrong**: Features derived from target won't be available at prediction time

4. **Wrong Cross-Validation (Line 152)**
   - **Issue**: Ran cross-validation on test set instead of training set
   - **Impact**: Invalid model validation, defeats purpose of CV
   - **Why it's wrong**: CV should only be used on training data

### Other Issues:

5. **Poor Missing Value Handling (Line 77)**
   - **Issue**: Filled all NaN with 0 without analysis
   - **Impact**: May introduce bias or incorrect values
   - **Better approach**: Analyze missing patterns, use appropriate imputation

6. **Incomplete Evaluation**
   - **Issue**: Only used accuracy metric
   - **Impact**: Incomplete understanding of model performance
   - **Better approach**: Use Precision, Recall, F1, ROC-AUC

7. **No Random State**
   - **Issue**: `train_test_split` without `random_state` parameter
   - **Impact**: Results not reproducible
   - **Better approach**: Always set `random_state` for reproducibility

---

## ✅ Corrections Applied

### 1. Removed All Leakage Features
```python
# ❌ WRONG (Broken notebook)
df['attrition_copy'] = df['attrition']  # Direct target copy
df['target_leakage_feature'] = ...      # Derived from target

# ✅ CORRECT
X = df.drop('attrition', axis=1)  # Only legitimate features
y = df['attrition']
```

### 2. Proper Train/Test Split Order
```python
# ❌ WRONG (Broken notebook)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ CORRECT
X_train, X_test = train_test_split(X, ...)  # Split FIRST
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training only
X_test_scaled = scaler.transform(X_test)  # Transform test
```

### 3. Proper Cross-Validation
```python
# ❌ WRONG (Broken notebook)
cv_scores = cross_val_score(model, X_test, y_test, cv=5)

# ✅ CORRECT
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
```

### 4. Comprehensive Evaluation
```python
# ✅ Multiple metrics for complete picture
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}
```

### 5. Reproducibility
```python
# ✅ Set random state everywhere
RANDOM_STATE = 42
train_test_split(..., random_state=RANDOM_STATE)
RandomForestClassifier(..., random_state=RANDOM_STATE)
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

1. **Navigate to the directory:**
   ```bash
   cd assignment-2-leakage-debugging
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open `assignment-2-debugging.ipynb`**

4. **Run all cells** (Cell → Run All)

### Expected Runtime
- Approximately 2-3 minutes on a standard laptop

---

## 📊 Expected Results

### Broken Model (with leakage):
- **Accuracy**: ~99%+ (unrealistically high)
- **ROC-AUC**: ~0.99+ (near perfect)
- **Reality**: Will FAIL in production

### Corrected Model (realistic):
- **Accuracy**: ~70-85% (realistic range)
- **ROC-AUC**: ~0.70-0.85 (realistic range)
- **Reality**: Will generalize to production

### Key Insight
The broken model's performance is **artificially inflated** due to data leakage. The corrected model shows **realistic performance** that will actually generalize to new data.

---

## 🎁 Extra Features Implemented

### 1. sklearn Pipeline
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```
**Benefits:**
- Prevents data leakage automatically
- Cleaner, more maintainable code
- Easier to deploy to production

### 2. Data Leakage Detection Utility
```python
def detect_data_leakage(X, y, threshold=0.95):
    """Detect features highly correlated with target"""
    # Flags suspicious features that may be leaking information
```
**Benefits:**
- Automatically detects potential leakage
- Checks feature-target correlations
- Early warning system for data issues

### 3. Comprehensive Visualizations
- Performance comparison charts
- Feature importance analysis
- Confusion matrices

---

## 📚 Key Takeaways

### Best Practices for ML Pipelines:

1. **✅ Always split data FIRST** before any preprocessing
2. **✅ Fit preprocessing only on training data**
3. **✅ Never use target-derived features**
4. **✅ Use sklearn Pipeline** to prevent leakage
5. **✅ Validate with multiple metrics**, not just accuracy
6. **✅ Check feature correlations** with target before modeling
7. **✅ Set random_state** for reproducibility
8. **✅ Cross-validate on training data only**

### Common Data Leakage Scenarios:

1. **Target Leakage**: Features that directly contain target information
2. **Temporal Leakage**: Using future information to predict the past
3. **Preprocessing Leakage**: Fitting on entire dataset before splitting
4. **Feature Engineering Leakage**: Creating features using test set information

---

## 🐛 Debugging Report

### Summary of All Corrections

| Issue | Location | Severity | Fix |
|-------|----------|----------|-----|
| Target copy feature | Line 67 | 🔴 Critical | Removed `attrition_copy` |
| Leakage feature | Line 70 | 🔴 Critical | Removed `target_leakage_feature` |
| Scaling before split | Lines 99-106 | 🔴 Critical | Split first, then scale |
| Wrong CV usage | Line 152 | 🔴 Critical | CV on training set only |
| Poor imputation | Line 77 | ⚠️ Medium | Analyzed missing values first |
| Single metric | Lines 129-131 | ⚠️ Medium | Added all metrics |
| No random state | Line 106 | ⚠️ Low | Added `random_state=42` |

---

## 📈 Performance Comparison

### Metrics Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
|-------|----------|-----------|--------|----------|---------|--------|
| Broken Pipeline (RF) | ~0.99 | ~0.99 | ~0.99 | ~0.99 | ~0.99 | ❌ LEAKAGE |
| Corrected Pipeline (RF) | ~0.75-0.85 | ~0.60-0.80 | ~0.50-0.70 | ~0.55-0.75 | ~0.70-0.85 | ✅ CLEAN |
| Corrected Pipeline (LR) | ~0.70-0.80 | ~0.55-0.75 | ~0.45-0.65 | ~0.50-0.70 | ~0.65-0.80 | ✅ CLEAN |

*Note: Exact values depend on random split and data characteristics*

---

## 🔍 How to Detect Data Leakage

### Manual Checks:

1. **Feature Analysis**
   - Check if any features are derived from target
   - Look for suspiciously high correlations (>0.95)
   - Verify temporal ordering of features

2. **Performance Checks**
   - If accuracy is too good to be true (~99%+), investigate
   - Compare train vs test performance
   - Check if performance drops dramatically in production

3. **Code Review**
   - Ensure split happens before preprocessing
   - Verify CV is only on training data
   - Check feature engineering logic

### Automated Checks:

Use the provided `detect_data_leakage()` utility function to automatically flag suspicious features.

---

## 📝 Documentation

### Setup Steps
1. Install required packages (see Prerequisites)
2. Ensure `attrition.csv` is in `../assignment-1-attrition/` directory
3. Open and run the corrected notebook

### Key Decisions
- **Dataset**: Using same attrition data as Assignment 1
- **Models**: Random Forest and Logistic Regression for comparison
- **Metrics**: All standard classification metrics for comprehensive evaluation
- **Pipeline**: Implemented sklearn Pipeline as best practice

### Future Improvements
- Add more sophisticated imputation methods
- Implement SMOTE for class imbalance
- Add hyperparameter tuning with GridSearchCV
- Create interactive dashboard for results

---

## 🎓 Learning Outcomes

After completing this assignment, you should understand:

1. **What is data leakage** and why it's dangerous
2. **How to identify** common leakage patterns
3. **How to prevent** leakage in ML pipelines
4. **Why proper workflow order** matters (split → fit → transform)
5. **How to use sklearn Pipeline** for clean, leakage-free code
6. **Why high accuracy** doesn't always mean a good model

---

## 📞 Questions?

If you encounter any issues or have questions:
1. Review the debugging report section
2. Check the key takeaways for best practices
3. Compare your code with the corrected notebook
4. Use the leakage detection utility to check your features

---

## ✅ Assignment Checklist

- [x] Identified all issues in broken notebook
- [x] Explained each type of data leakage
- [x] Implemented corrected pipeline
- [x] Compared before/after performance
- [x] Added sklearn Pipeline implementation
- [x] Created leakage detection utility
- [x] Comprehensive documentation
- [x] Visualizations and comparisons

---

**Assignment Status: ✅ COMPLETE**

This implementation demonstrates professional ML engineering practices and serves as a reference for avoiding data leakage in future projects.
