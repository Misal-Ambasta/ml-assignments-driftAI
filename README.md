# ML Assignments - DriftAI

Complete machine learning assignment solutions demonstrating data analysis, debugging, and feature engineering skills.

---

## 📚 Assignments Overview

### ✅ Assignment 1 - Employee Attrition Prediction
**Status**: Complete  
**Objective**: Build a classification model to predict employee attrition

**Key Features**:
- Binary classification problem
- Multiple models (Logistic Regression, Random Forest, etc.)
- Comprehensive evaluation metrics
- Feature importance analysis
- SHAP values for interpretability

**Location**: `assignment-1-attrition/`

---

### ✅ Assignment 2 - ML Pipeline Debugging & Data Leakage Detection
**Status**: Complete  
**Objective**: Identify and fix data leakage issues in a broken ML pipeline

**Key Features**:
- 7 critical issues identified and fixed
- Before/after performance comparison
- sklearn Pipeline implementation
- Data leakage detection utility
- Comprehensive debugging report

**Issues Fixed**:
1. Target leakage (direct copy of target)
2. Scaling before train/test split
3. Suspicious leakage features
4. Wrong cross-validation usage
5. Poor missing value handling
6. Incomplete evaluation metrics
7. No reproducibility (missing random_state)

**Performance Impact**:
- Broken model: ~99% accuracy (unrealistic, will fail in production)
- Corrected model: ~75-85% accuracy (realistic, will generalize)

**Location**: `assignment-2-leakage-debugging/`

---

### ✅ Assignment 3 - Productivity Feature Engineering & Optimization
**Status**: Complete  
**Objective**: Predict employee productivity through advanced feature engineering

**Key Features**:
- 15+ engineered features
- Baseline vs optimized comparison
- K-Means clustering for behavioral segments
- PCA for dimensionality reduction
- Feature importance dashboard
- Hyperparameter tuning

**Performance Improvement**:
- Baseline R²: ~0.20
- Optimized R²: ~0.58
- **Improvement: +190% (2.9x better)**

**Location**: `assignment-3-productivity/`

---

## 🚀 Quick Start

### Prerequisites


```bash
# Install dependencies using either uv or pip
# With uv (recommended):
uv pip install -r requirements.txt
# or, if using pyproject.toml:
uv pip install

# With pip:
pip install -r requirements.txt
```
See requirements.txt for the full list of packages.

### Running Assignments

Each assignment has its own directory with:
- Jupyter notebook (`.ipynb`)
- Dataset (`.csv`)
- Documentation (`README.md`)

**To run any assignment:**

```bash
# Navigate to assignment directory
cd assignment-X-name

# Start Jupyter Notebook
jupyter notebook

# Open the .ipynb file and run all cells
```

---

## 📁 Project Structure

```
ml-assignments-driftAI/
│
├── assignment-1-attrition/
│   ├── assignment-1-attrition.ipynb
│   ├── attrition.csv
│   └── README.md
│
├── assignment-2-leakage-debugging/
│   ├── assignment-2-debugging.ipynb      # ✅ Corrected version
│   ├── debug_broken_notebook.ipynb       # ❌ Broken version (reference)
│   ├── README.md
│   └── DEBUGGING_REPORT.md
│
├── assignment-3-productivity/
│   ├── assignment-3-productivity.ipynb
│   ├── employee_productivity.csv
│   └── README.md
│
├── problem-statements.md
└── README.md (this file)
```

---

## 🎯 Skills Demonstrated

### Machine Learning
- ✅ Classification (Logistic Regression, Random Forest, etc.)
- ✅ Regression (Linear, Ridge, Gradient Boosting)
- ✅ Ensemble methods
- ✅ Hyperparameter tuning
- ✅ Cross-validation

### Feature Engineering
- ✅ Domain-specific features
- ✅ Interaction terms
- ✅ Polynomial features
- ✅ Categorical encoding
- ✅ Feature selection

### Data Analysis
- ✅ EDA (Exploratory Data Analysis)
- ✅ Correlation analysis
- ✅ Missing value handling
- ✅ Outlier detection
- ✅ Data visualization

### Advanced Techniques
- ✅ K-Means clustering
- ✅ PCA (Principal Component Analysis)
- ✅ SHAP values
- ✅ Feature importance analysis
- ✅ Data leakage detection

### Best Practices
- ✅ Proper train/test split
- ✅ sklearn Pipeline usage
- ✅ Reproducibility (random_state)
- ✅ Comprehensive evaluation metrics
- ✅ Documentation

---

## 📊 Results Summary

| Assignment | Task | Best Model | Performance | Improvement |
|------------|------|------------|-------------|-------------|
| 1 | Attrition Prediction | Random Forest | ~85% Accuracy | Baseline |
| 2 | Leakage Debugging | Random Forest | ~80% Accuracy | Fixed from 99% (leakage) |
| 3 | Productivity Prediction | Gradient Boosting | R² = 0.58 | +190% from baseline |

---

## 🎓 Key Learnings

### Assignment 1: Classification
- Handling imbalanced datasets
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance for business insights
- Model interpretability with SHAP

### Assignment 2: Debugging
- **Critical**: Always split data BEFORE preprocessing
- Identify data leakage patterns
- Use sklearn Pipeline to prevent leakage
- If accuracy is ~99%+, investigate for leakage
- Cross-validate on training data only

### Assignment 3: Feature Engineering
- Feature engineering can improve performance by 190%+
- Domain knowledge is crucial for creating meaningful features
- Clustering reveals behavioral patterns
- Feature selection prevents overfitting
- Visualizations make insights actionable

---

## 🔧 Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations

### Development
- **Jupyter Notebook**: Interactive development
- **Python 3.8+**: Programming language

---

## 📝 Documentation

Each assignment includes:
- ✅ **Jupyter Notebook**: Complete implementation with explanations
- ✅ **README.md**: Setup instructions and methodology
- ✅ **Code Comments**: Detailed inline documentation
- ✅ **Visualizations**: Charts and plots for insights

**Assignment 2 also includes**:
- ✅ **DEBUGGING_REPORT.md**: Detailed analysis of all issues

---

## 🎯 Assignment Requirements Met

### Assignment 1 ✅
- [x] Load and prepare dataset
- [x] Perform EDA
- [x] Train multiple classification models
- [x] Evaluate with all metrics
- [x] Identify top attrition drivers
- [x] **EXTRA**: SHAP values
- [x] **EXTRA**: Hyperparameter tuning

### Assignment 2 ✅
- [x] Review broken notebook
- [x] Identify all issues (7 found)
- [x] Explain data leakage
- [x] Correct preprocessing and splitting
- [x] Retrain clean pipeline
- [x] Compare before vs after
- [x] **EXTRA**: sklearn Pipeline
- [x] **EXTRA**: Leakage detection utility
- [x] **EXTRA**: Comprehensive visualizations

### Assignment 3 ✅
- [x] Load and clean dataset
- [x] Build baseline regression model
- [x] Engineer multiple new features (15+)
- [x] Apply scaling and feature selection
- [x] Build optimized model
- [x] Compare baseline vs optimized
- [x] **EXTRA**: Clustering-based features
- [x] **EXTRA**: PCA analysis
- [x] **EXTRA**: Feature importance dashboard

---

## 💡 Best Practices Demonstrated

1. **Data Splitting**
   ```python
   # ✅ CORRECT: Split first, then preprocess
   X_train, X_test = train_test_split(X, y, ...)
   scaler.fit(X_train)
   ```

2. **Reproducibility**
   ```python
   # ✅ Always set random_state
   RANDOM_STATE = 42
   train_test_split(..., random_state=RANDOM_STATE)
   ```

3. **Pipeline Usage**
   ```python
   # ✅ Prevents data leakage
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', RandomForestClassifier())
   ])
   ```

4. **Comprehensive Evaluation**
   ```python
   # ✅ Multiple metrics, not just accuracy
   metrics = {
       'Accuracy': accuracy_score(...),
       'Precision': precision_score(...),
       'Recall': recall_score(...),
       'F1': f1_score(...),
       'ROC-AUC': roc_auc_score(...)
   }
   ```

---

## 🐛 Common Issues & Solutions

### Issue: "Module not found"
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Issue: "File not found"
```bash
# Ensure you're in the correct directory
cd assignment-X-name
```

### Issue: "Kernel keeps dying"
```python
# Reduce model complexity
# Change n_estimators from 200 to 50
```

---

## 📈 Performance Metrics Explained

### Classification Metrics (Assignment 1, 2)
- **Accuracy**: Overall correctness (use with caution for imbalanced data)
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many did we catch
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (threshold-independent)

### Regression Metrics (Assignment 3)
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAE**: Mean Absolute Error (average error magnitude)

---

## 🎓 Learning Path

**Recommended Order**:
1. **Assignment 1**: Learn classification basics
2. **Assignment 2**: Understand data leakage (critical!)
3. **Assignment 3**: Master feature engineering

---

## 📚 Additional Resources

### Concepts Covered
- Train/test split
- Cross-validation
- Feature engineering
- Data leakage
- Model evaluation
- Hyperparameter tuning
- Clustering
- Dimensionality reduction
- Model interpretability

### Further Reading
- scikit-learn documentation
- Feature Engineering for Machine Learning
- Hands-On Machine Learning with Scikit-Learn
- Interpretable Machine Learning

---

## ✅ Project Status

| Assignment | Status | Notebook | Documentation | Extras |
|------------|--------|----------|---------------|--------|
| Assignment 1 | ✅ Complete | ✅ | ✅ | ✅ |
| Assignment 2 | ✅ Complete | ✅ | ✅ | ✅ |
| Assignment 3 | ✅ Complete | ✅ | ✅ | ✅ |

**Overall Status**: ✅ **ALL ASSIGNMENTS COMPLETE**

---

## 🎯 Key Achievements

1. ✅ **3 complete ML projects** with production-ready code
2. ✅ **15+ engineered features** demonstrating domain expertise
3. ✅ **7 critical bugs** identified and fixed in Assignment 2
4. ✅ **190% performance improvement** in Assignment 3
5. ✅ **Comprehensive documentation** for all assignments
6. ✅ **Extra features** implemented in all assignments
7. ✅ **Best practices** followed throughout

---

## 🎉 Conclusion

This repository demonstrates comprehensive machine learning skills including:
- Data analysis and preprocessing
- Model training and evaluation
- Debugging and error detection
- Advanced feature engineering
- Production-ready code practices
- Clear documentation

All assignments meet the basic requirements and include extra features for enhanced learning and demonstration of advanced skills.

---


*Happy Learning! 🚀*