# Debugging Report - Assignment 2
# ML Pipeline Data Leakage Detection

**Date**: December 15, 2025  
**Assignment**: ML Pipeline Debugging & Data Leakage Detection  
**Reviewed Notebook**: `debug_broken_notebook.ipynb`  
**Status**: ❌ CRITICAL ISSUES FOUND - DO NOT USE IN PRODUCTION

---

## Executive Summary

The reviewed notebook contains **7 critical and medium-severity issues** that result in severe data leakage and artificially inflated model performance. The model shows near-perfect accuracy (~99%+) during testing but will **fail catastrophically in production** due to these issues.

**Recommendation**: Use the corrected implementation in `assignment-2-debugging.ipynb` instead.

---

## Detailed Issue Analysis

### 🔴 CRITICAL ISSUE #1: Target Leakage via Feature Copy

**Location**: Line 67  
**Severity**: CRITICAL  
**Code**:
```python
df["attrition_copy"] = df["attrition"]  # <-- target copied into features
```

**Problem**:
- Creates a feature that is an exact copy of the target variable
- Model has direct access to the answer during training and testing
- This is the most blatant form of data leakage

**Impact**:
- Accuracy inflated to ~100%
- Model learns to simply look at this feature instead of finding real patterns
- Feature will not exist in production data → model will fail completely

**Why It Happens**:
- Sometimes developers create helper columns during EDA and forget to remove them
- May occur when joining datasets and accidentally including target from another source

**Correct Approach**:
```python
# Remove target from features
X = df.drop('attrition', axis=1)  # Only legitimate features
y = df['attrition']  # Target separate
```

---

### 🔴 CRITICAL ISSUE #2: Scaling Before Train/Test Split

**Location**: Lines 99-106  
**Severity**: CRITICAL  
**Code**:
```python
# Scale ALL data before splitting (data leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fits on entire dataset

# Split AFTER scaling (this uses information from the full dataset)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

**Problem**:
- StandardScaler calculates mean and standard deviation from entire dataset
- Test set statistics (mean, std) leak into training data
- Model indirectly sees information from test set during training

**Impact**:
- Model performance appears better than it actually is
- Overestimates generalization ability
- In production, scaler won't have access to test statistics → performance drop

**Example of Leakage**:
```
Original data: [1, 2, 3, 100]  # 100 is in test set
After scaling with full data: [-0.68, -0.65, -0.62, 1.95]
# Training data now "knows" there's a large value (100) in test set
```

**Correct Approach**:
```python
# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test using train stats
```

---

### 🔴 CRITICAL ISSUE #3: Suspicious Leakage Feature

**Location**: Line 70  
**Severity**: CRITICAL  
**Code**:
```python
# 2) Use 'target_leakage_feature' as-is assuming it's a good predictor
# (We don't check how this was created.)
```

**Problem**:
- Feature named `target_leakage_feature` is used without investigation
- Likely derived from or highly correlated with target variable
- No validation of feature legitimacy

**Impact**:
- Hidden data leakage
- Model relies on information that won't be available at prediction time
- Artificially inflated performance metrics

**How to Detect**:
```python
# Check correlation with target
correlation = df['target_leakage_feature'].corr(df['attrition'])
if abs(correlation) > 0.95:
    print("⚠️ WARNING: Potential data leakage!")
```

**Correct Approach**:
- Investigate origin of every feature
- Check correlations with target
- Remove features derived from target
- Document feature engineering process

---

### 🔴 CRITICAL ISSUE #4: Cross-Validation on Test Set

**Location**: Line 152  
**Severity**: CRITICAL  
**Code**:
```python
# Run cross-validation only on the TEST set (incorrect!)
cv_scores = cross_val_score(model, X_test, y_test, cv=5)
print("Cross-validation scores on test set:", cv_scores)
```

**Problem**:
- Cross-validation should ONLY be performed on training data
- Running CV on test set defeats the entire purpose of having a test set
- Test set should remain completely untouched until final evaluation

**Impact**:
- Invalid model validation
- No true holdout set for unbiased evaluation
- Overly optimistic performance estimates

**What CV Should Do**:
- Split training data into K folds
- Train on K-1 folds, validate on 1 fold
- Repeat K times
- Average performance across folds

**Correct Approach**:
```python
# CV on TRAINING set only
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores on training set:", cv_scores)

# Test set used ONLY for final evaluation
final_score = model.score(X_test, y_test)
```

---

### ⚠️ MEDIUM ISSUE #5: Poor Missing Value Handling

**Location**: Line 77  
**Severity**: MEDIUM  
**Code**:
```python
# Fill all missing values with 0 without analysis
df = df.fillna(0)
```

**Problem**:
- Fills all missing values with 0 without understanding why they're missing
- No analysis of missing data patterns
- May introduce bias or incorrect values

**Impact**:
- For some features, 0 might be a valid value, creating confusion
- For others (like income), 0 is clearly wrong
- May bias model predictions

**Better Approaches**:
```python
# 1. Analyze missing patterns first
print(df.isnull().sum())

# 2. Use appropriate imputation
from sklearn.impute import SimpleImputer

# For numerical: use median or mean
num_imputer = SimpleImputer(strategy='median')

# For categorical: use most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')

# 3. Or use more sophisticated methods
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
```

---

### ⚠️ MEDIUM ISSUE #6: Incomplete Evaluation Metrics

**Location**: Lines 129-131  
**Severity**: MEDIUM  
**Code**:
```python
# Evaluate using Accuracy only
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
```

**Problem**:
- Only uses accuracy metric
- Accuracy can be misleading, especially with imbalanced datasets
- Missing important metrics like precision, recall, F1, ROC-AUC

**Impact**:
- Incomplete understanding of model performance
- May miss important trade-offs (precision vs recall)
- Can't assess model's behavior on minority class

**Example of Accuracy Paradox**:
```
Dataset: 95% class 0, 5% class 1
Model that always predicts 0: 95% accuracy!
But it never detects class 1 (0% recall on minority class)
```

**Correct Approach**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\n", classification_report(y_test, y_pred))
```

---

### ⚠️ LOW ISSUE #7: No Random State

**Location**: Line 106  
**Severity**: LOW  
**Code**:
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# No random_state parameter
```

**Problem**:
- Results are not reproducible
- Each run produces different train/test splits
- Makes debugging and comparison difficult

**Impact**:
- Can't reproduce results
- Hard to compare different models fairly
- Difficult to debug issues

**Correct Approach**:
```python
RANDOM_STATE = 42  # Or any fixed integer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y  # Bonus: maintain class distribution
)

# Also set in models
model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)
```

---

## Performance Impact Analysis

### Broken Pipeline Results (Estimated):
```
Accuracy:  0.99+  ← Unrealistically high
Precision: 0.99+
Recall:    0.99+
F1 Score:  0.99+
ROC-AUC:   0.99+
```

**Why so high?**
- Target leakage features give away the answer
- Model essentially "cheats" by looking at target-derived features

### Corrected Pipeline Results (Realistic):
```
Accuracy:  0.70-0.85  ← Realistic range
Precision: 0.60-0.80
Recall:    0.50-0.70
F1 Score:  0.55-0.75
ROC-AUC:   0.70-0.85
```

**Why lower?**
- No leakage → model must find real patterns
- These results will actually generalize to production

### Production Impact:
- **Broken model**: Will fail completely (accuracy may drop to ~50% or worse)
- **Corrected model**: Will maintain performance similar to test results

---

## Recommendations

### Immediate Actions:
1. ❌ **DO NOT use** `debug_broken_notebook.ipynb` for any production purposes
2. ✅ **USE** `assignment-2-debugging.ipynb` as the correct implementation
3. 🔍 **REVIEW** all existing models for similar leakage issues

### Best Practices Going Forward:

#### 1. Data Splitting
```python
# ALWAYS split before any preprocessing
X_train, X_test = train_test_split(X, y, ...)
# THEN fit preprocessing on training data only
scaler.fit(X_train)
```

#### 2. Feature Engineering
```python
# Check every feature's origin
# Remove any feature derived from target
# Verify temporal ordering (no future information)
```

#### 3. Use sklearn Pipeline
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
# Pipeline handles preprocessing correctly within CV
```

#### 4. Comprehensive Evaluation
```python
# Always use multiple metrics
# Especially important for imbalanced datasets
# Include confusion matrix and classification report
```

#### 5. Reproducibility
```python
# Set random_state everywhere
# Document all random seeds
# Version control your code
```

---

## Leakage Detection Checklist

Use this checklist for all future ML projects:

- [ ] Split data BEFORE any preprocessing
- [ ] Fit preprocessing only on training data
- [ ] Check all features for correlation with target (>0.95 is suspicious)
- [ ] Verify no features are derived from target
- [ ] Ensure temporal ordering (no future information)
- [ ] Cross-validate only on training data
- [ ] Use multiple evaluation metrics
- [ ] Set random_state for reproducibility
- [ ] Test on truly unseen data before production
- [ ] Monitor production performance vs test performance

---

## Conclusion

The broken notebook contains **severe data leakage issues** that make it completely unsuitable for production use. While it shows impressive test metrics (~99% accuracy), these results are **artificially inflated** and will not generalize.

The corrected implementation demonstrates:
- ✅ Proper train/test split workflow
- ✅ No data leakage
- ✅ Realistic performance metrics
- ✅ Production-ready code
- ✅ Best practices for ML pipelines

**Final Verdict**: 
- Broken notebook: ❌ REJECT
- Corrected notebook: ✅ APPROVED

---

## Appendix: Quick Reference

### Data Leakage Types

1. **Target Leakage**: Features that contain target information
2. **Train-Test Contamination**: Test data used in training
3. **Temporal Leakage**: Future information used to predict past
4. **Preprocessing Leakage**: Fitting on entire dataset before split

### Common Leakage Scenarios

| Scenario | Example | Fix |
|----------|---------|-----|
| Target in features | `df['target_copy'] = df['target']` | Remove feature |
| Scaling before split | `scaler.fit(X)` then split | Split first, then scale |
| CV on test set | `cross_val_score(model, X_test, y_test)` | CV on training only |
| Future information | Using next month's data to predict this month | Ensure temporal ordering |

### Detection Methods

1. **Too-good-to-be-true performance** (>95% accuracy)
2. **High feature-target correlation** (>0.95)
3. **Large train-test performance gap**
4. **Production performance much worse than test**

---

**Report Prepared By**: ML Engineering Team  
**Review Status**: COMPLETE  
**Next Steps**: Implement corrected pipeline and monitor production performance
