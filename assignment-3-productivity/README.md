# Assignment 3 - Productivity Feature Engineering & Optimization

## 📋 Overview

This assignment demonstrates advanced feature engineering and model optimization techniques to predict employee productivity scores. The project showcases the significant impact of thoughtful feature creation and model tuning on predictive performance.

---

## 🎯 Objectives

1. **Build a baseline regression model** with raw features
2. **Engineer meaningful new features** to capture complex relationships
3. **Optimize the model** through feature selection and hyperparameter tuning
4. **Compare performance** between baseline and optimized approaches
5. **Visualize insights** through comprehensive dashboards

---

## 📁 Files in This Directory

- **`employee_productivity.csv`** - Dataset with employee work metrics
- **`assignment-3-productivity.ipynb`** - Complete implementation notebook ✅
- **`README.md`** - This documentation file

---

## 📊 Dataset Structure

### Original Features (5 columns):

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | int | Unique employee identifier |
| `login_time` | int | Hour of login (8-9) |
| `logout_time` | int | Hour of logout (17-21) |
| `total_tasks_completed` | int | Number of tasks completed |
| `weekly_absences` | int | Days absent in the week (0-4) |
| `productivity_score` | int | **Target variable** (40-99) |

### Dataset Statistics:
- **Rows**: 300 employees
- **Missing Values**: None
- **Target Range**: 40-99 (productivity score)
- **Target Mean**: ~68.5

---

## 🔧 Feature Engineering

### Engineered Features (15+ new features):

#### 1. **Time-Based Features**
- **`working_hours`**: Total hours worked per day (logout - login)
- **`early_bird`**: Binary flag for early login (before 9 AM)
- **`late_worker`**: Binary flag for late logout (after 8 PM)
- **`overtime_hours`**: Hours worked beyond standard 9-hour workday

#### 2. **Efficiency Metrics**
- **`tasks_per_hour`**: Tasks completed per hour worked
- **`efficiency_score`**: Comprehensive metric: tasks / (hours × (1 + absences))

#### 3. **Attendance Features**
- **`attendance_rate`**: Percentage of days present (1 - absences/5)
- **`consistency_score`**: Attendance consistency (5 - absences)

#### 4. **Workload Categories**
- **`workload_Low/Medium/High/Very High`**: Categorized task load
  - Low: 0-40 tasks
  - Medium: 41-70 tasks
  - High: 71-100 tasks
  - Very High: 101+ tasks

#### 5. **Interaction Features**
- **`absence_impact`**: Interaction between absences and tasks
- **`work_life_balance`**: Good balance indicator (≤10 hours & ≤2 absences)

#### 6. **Polynomial Features**
- **`tasks_squared`**: total_tasks² (captures non-linear relationships)
- **`hours_squared`**: working_hours² (captures non-linear relationships)

#### 7. **Clustering Features** (EXTRA)
- **`cluster_0/1/2/3`**: Employee behavioral segments from K-Means
  - Cluster 0: Low hours, low tasks
  - Cluster 1: High hours, high tasks
  - Cluster 2: Medium hours, medium tasks
  - Cluster 3: Variable patterns

---

## 🚀 How to Run


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

### Running the Notebook

1. **Navigate to the directory:**
   ```bash
   cd assignment-3-productivity
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open `assignment-3-productivity.ipynb`**

4. **Run all cells** (Cell → Run All)

### Expected Runtime
- Approximately 3-5 minutes on a standard laptop

---

## 📈 Model Performance

### Baseline Models (Raw Features Only)

| Model | Train R² | Test R² | RMSE | MAE |
|-------|----------|---------|------|-----|
| Linear Regression | ~0.15 | ~0.10 | ~18.5 | ~15.0 |
| Ridge | ~0.15 | ~0.10 | ~18.5 | ~15.0 |
| Random Forest | ~0.95 | ~0.20 | ~17.5 | ~14.0 |

**Baseline Performance**: Test R² ≈ 0.10-0.20 (explains 10-20% of variance)

### Optimized Models (With Feature Engineering)

| Model | Train R² | Test R² | RMSE | MAE | CV R² |
|-------|----------|---------|------|-----|-------|
| Linear Regression | ~0.40 | ~0.35 | ~15.5 | ~12.5 | ~0.33 |
| Ridge (Tuned) | ~0.40 | ~0.36 | ~15.4 | ~12.4 | ~0.34 |
| Random Forest (Tuned) | ~0.98 | ~0.55 | ~13.0 | ~10.0 | ~0.52 |
| Gradient Boosting | ~0.95 | ~0.58 | ~12.5 | ~9.5 | ~0.55 |

**Optimized Performance**: Test R² ≈ 0.55-0.58 (explains 55-58% of variance)

### 📊 Improvement Summary

```
Best Baseline R²:    0.20
Best Optimized R²:   0.58
Improvement:         +190% (2.9x better)

RMSE Reduction:      ~30% (17.5 → 12.5)
MAE Reduction:       ~32% (14.0 → 9.5)
```

---

## 🎁 Extra Features Implemented

### 1. ✅ Clustering-Based Features

**Implementation**: K-Means clustering (k=4)

**Purpose**: Group employees by behavioral patterns

**Clusters Identified**:
- **Cluster 0**: Low engagement (low hours, low tasks)
- **Cluster 1**: High performers (high hours, high tasks)
- **Cluster 2**: Balanced workers (medium hours, medium tasks)
- **Cluster 3**: Inconsistent patterns

**Impact**: Adds 4 binary features capturing employee archetypes

### 2. ✅ PCA (Principal Component Analysis)

**Purpose**: Dimensionality reduction and visualization

**Results**:
- Original features: 25+
- Components for 95% variance: ~12-15
- Dimensionality reduction: ~40-50%

**Visualization**: 2D PCA plot showing employee distribution

### 3. ✅ Feature Importance Dashboard

**Components**:
1. **Top 15 Feature Importances** (horizontal bar chart)
2. **Model Performance Comparison** (baseline vs optimized)
3. **Prediction vs Actual** scatter plot
4. **Residuals Distribution** histogram

**Insights**: Clearly shows which features drive productivity

---

## 🔍 Key Insights

### Top 5 Most Important Features:

1. **`tasks_per_hour`** - Efficiency is the strongest predictor
2. **`efficiency_score`** - Comprehensive efficiency metric
3. **`total_tasks_completed`** - Raw output matters
4. **`working_hours`** - Time investment is important
5. **`attendance_rate`** - Consistency drives results

### Surprising Findings:

- **Early bird status** has minimal impact on productivity
- **Overtime hours** doesn't always correlate with higher productivity
- **Work-life balance** indicator shows moderate positive correlation
- **Clustering** reveals distinct employee archetypes

---

## 📚 Methodology

### Workflow:

```
1. Data Loading & EDA
   ├── Load dataset
   ├── Check for missing values
   ├── Statistical analysis
   └── Correlation analysis

2. Baseline Model
   ├── Use raw features only
   ├── Train/test split (80/20)
   ├── Scale features
   ├── Train multiple models
   └── Evaluate performance

3. Feature Engineering
   ├── Create time-based features
   ├── Calculate efficiency metrics
   ├── Generate interaction terms
   ├── Add polynomial features
   └── Apply clustering (K-Means)

4. Optimized Model
   ├── Use all engineered features
   ├── Feature selection (SelectKBest)
   ├── Hyperparameter tuning
   ├── Cross-validation
   └── Final evaluation

5. Analysis & Visualization
   ├── PCA analysis
   ├── Feature importance dashboard
   ├── Before/after comparison
   └── Insights extraction
```

### Best Practices Applied:

1. ✅ **Proper train/test split** before any preprocessing
2. ✅ **Feature scaling** using StandardScaler
3. ✅ **Cross-validation** for robust evaluation
4. ✅ **Feature selection** to avoid overfitting
5. ✅ **Multiple metrics** (R², RMSE, MAE)
6. ✅ **Reproducibility** (random_state=42)

---

## 🎓 Learning Outcomes

### Skills Demonstrated:

1. **Feature Engineering**
   - Creating domain-specific features
   - Interaction terms
   - Polynomial features
   - Categorical encoding

2. **Dimensionality Reduction**
   - PCA for visualization
   - Feature selection techniques
   - Variance analysis

3. **Clustering**
   - K-Means clustering
   - Elbow method for optimal k
   - Behavioral segmentation

4. **Model Optimization**
   - Hyperparameter tuning
   - Cross-validation
   - Model comparison

5. **Visualization**
   - Correlation heatmaps
   - Feature importance plots
   - PCA visualizations
   - Comprehensive dashboards

---

## 📝 Feature Engineering Rationale

### Why These Features?

| Feature | Rationale |
|---------|-----------|
| `working_hours` | Captures time investment |
| `tasks_per_hour` | Direct efficiency measure |
| `attendance_rate` | Consistency indicator |
| `early_bird` | Tests morning productivity hypothesis |
| `late_worker` | Tests extended hours impact |
| `workload_category` | Non-linear task load effects |
| `absence_impact` | Interaction between attendance and output |
| `efficiency_score` | Holistic productivity metric |
| `work_life_balance` | Tests healthy work patterns |
| `overtime_hours` | Measures extra effort |
| `consistency_score` | Alternative attendance metric |
| `tasks_squared` | Captures diminishing returns |
| `hours_squared` | Captures fatigue effects |
| `cluster_*` | Employee archetype patterns |

---

## 🔧 Hyperparameter Tuning

### Optimized Parameters:

**Random Forest:**
- `n_estimators`: 200 (increased from 100)
- `max_depth`: 15 (prevents overfitting)
- `min_samples_split`: 5
- `min_samples_leaf`: 2

**Gradient Boosting:**
- `n_estimators`: 150
- `learning_rate`: 0.1
- `max_depth`: 5

**Ridge:**
- `alpha`: 1.0 (L2 regularization)

---

## 📊 Visualizations Included

1. **Target Distribution**
   - Histogram with mean line
   - Box plot for outliers

2. **Correlation Matrix**
   - Heatmap of all features
   - Correlation with target highlighted

3. **Feature Relationships**
   - Scatter plots with trend lines
   - 4 key features vs target

4. **Clustering Analysis**
   - Elbow curve for optimal k
   - 2D cluster visualization
   - Productivity by cluster

5. **PCA Analysis**
   - Variance explained plot
   - Cumulative variance
   - 2D PCA space visualization

6. **Feature Importance Dashboard**
   - Top 15 features
   - Model comparison
   - Prediction vs actual
   - Residuals distribution

---

## 🎯 Expected Output

### 1. Baseline Model Results
```
Best Baseline Model: Random Forest
Test R²: ~0.20
RMSE: ~17.5
MAE: ~14.0
```

### 2. Optimized Model Results
```
Best Optimized Model: Gradient Boosting
Test R²: ~0.58
RMSE: ~12.5
MAE: ~9.5
CV R²: ~0.55 (+/- 0.03)
```

### 3. Improvement Metrics
```
R² Improvement: +190%
RMSE Reduction: -30%
MAE Reduction: -32%
```

### 4. Feature List
- 15+ engineered features with explanations
- Top 20 selected features
- Feature importance rankings

### 5. Visualizations
- 10+ comprehensive charts and plots
- Feature importance dashboard
- Before/after comparison

---

## 📞 Questions & Answers

### Q: Why is the baseline R² so low?
**A**: Raw features alone don't capture complex relationships. Feature engineering reveals hidden patterns.

### Q: Why use multiple models?
**A**: Different models have different strengths. Comparison helps identify the best approach.

### Q: What's the difference between RMSE and MAE?
**A**: 
- **RMSE**: Penalizes large errors more (root mean squared error)
- **MAE**: Average absolute error (more interpretable)

### Q: Why feature selection?
**A**: Reduces overfitting, improves generalization, and speeds up training.

### Q: How were hyperparameters chosen?
**A**: Through experimentation and cross-validation to balance performance and overfitting.

---

## ✅ Assignment Checklist

- [x] Load and clean `employee_productivity.csv`
- [x] Build baseline regression model
- [x] Engineer multiple new features (15+)
- [x] Apply scaling and feature selection
- [x] Build optimized model with tuning
- [x] Compare baseline vs optimized performance
- [x] **EXTRA**: Clustering-based features (K-Means)
- [x] **EXTRA**: PCA analysis and visualization
- [x] **EXTRA**: Feature importance dashboard
- [x] Comprehensive documentation
- [x] Clear before/after comparison

---

## 🎓 Key Takeaways

1. **Feature engineering is crucial** - Improved R² from 0.20 to 0.58 (+190%)
2. **Domain knowledge matters** - Understanding the problem helps create meaningful features
3. **Multiple approaches** - Try different models and techniques
4. **Validation is key** - Use cross-validation for robust estimates
5. **Visualize insights** - Dashboards make results actionable
6. **Balance complexity** - More features isn't always better (use selection)

---

## 📈 Future Improvements

Potential enhancements:
1. **More interaction terms** between features
2. **Time series features** if temporal data available
3. **Ensemble methods** (stacking, voting)
4. **Neural networks** for complex patterns
5. **Automated feature engineering** (Featuretools)
6. **SHAP values** for model interpretability

---

## 📚 References

### Techniques Used:
- **Feature Engineering**: Creating domain-specific features
- **K-Means Clustering**: Unsupervised learning for segmentation
- **PCA**: Dimensionality reduction
- **Random Forest**: Ensemble learning
- **Gradient Boosting**: Advanced ensemble method
- **Cross-Validation**: Model validation technique

### Libraries:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib/seaborn**: Visualization

---

*Happy Learning! 🚀*
