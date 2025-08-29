# 🏠 Boston Housing Price Predictor

**Level**: 🟢 Beginner  
**Type**: Regression  
**Dataset**: Boston Housing Dataset (Scikit-learn built-in)

## 📋 Project Overview

This project predicts house prices in Boston using machine learning regression techniques. It's an excellent introduction to regression analysis, feature importance, and model evaluation metrics specific to continuous target variables. The Boston Housing dataset is a classic dataset in machine learning education.

## 🎯 Objectives

- Learn regression analysis fundamentals
- Understand feature importance in price prediction
- Practice regression evaluation metrics (MAE, MSE, R²)
- Compare different regression algorithms
- Visualize relationships between features and target
- Handle multicollinearity and feature selection

## 📊 Dataset Information

The Boston Housing dataset contains information about housing in the area of Boston, Massachusetts.

### Features (Input Variables)
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: Proportion of blacks by town
- **LSTAT**: % lower status of the population

### Target Variable
- **MEDV**: Median value of owner-occupied homes in $1000s

### Dataset Statistics
- **Total Samples**: 506 housing records
- **Features**: 13 numerical features
- **Target Range**: $5,000 - $50,000 (in 1970s dollars)
- **Missing Values**: None

## 🔍 Project Structure

```
03-boston-housing/
├── notebook.ipynb          # Main Jupyter notebook
├── README.md              # This file
├── requirements.txt       # Project dependencies
└── results/              # Generated plots and results
    ├── feature_correlations.png
    ├── price_distributions.png
    ├── model_comparison.png
    └── residual_plots.png
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Installation
1. Navigate to the project directory:
```bash
cd 03-boston-housing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

## 📈 Methodology

### 1. Data Loading & Exploration
- Load Boston Housing dataset from scikit-learn
- Explore dataset structure and statistics
- Analyze target variable distribution
- Check for outliers and data quality

### 2. Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Price distribution visualization
- Scatter plots for key relationships
- Geographic and demographic insights
- Feature importance preliminary analysis

### 3. Data Preprocessing
- Feature scaling and normalization
- Outlier detection and handling
- Feature selection techniques
- Train-test split with proper validation

### 4. Model Building & Training
- **Linear Regression**: Baseline linear model
- **Ridge Regression**: L2 regularization for overfitting
- **Lasso Regression**: L1 regularization with feature selection
- **Random Forest**: Ensemble method for non-linear relationships
- **Gradient Boosting**: Advanced ensemble technique
- **Support Vector Regression**: Non-linear regression with kernels

### 5. Model Evaluation
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) score
- Cross-validation analysis
- Residual analysis and plots

### 6. Model Interpretation
- Feature importance analysis
- Coefficient interpretation for linear models
- Residual plots for model diagnostics
- Prediction vs actual value analysis

## 📊 Expected Results

### Model Performance (Expected)
| Model | MAE | RMSE | R² Score | CV Score |
|-------|-----|------|----------|----------|
| Linear Regression | ~3.2 | ~4.7 | ~0.67 | ~0.64 |
| Ridge Regression | ~3.1 | ~4.6 | ~0.68 | ~0.66 |
| Lasso Regression | ~3.0 | ~4.5 | ~0.69 | ~0.67 |
| Random Forest | ~2.8 | ~4.1 | ~0.74 | ~0.71 |
| Gradient Boosting | ~2.7 | ~3.9 | ~0.76 | ~0.73 |
| SVR | ~3.0 | ~4.4 | ~0.70 | ~0.68 |

### Key Insights
- Number of rooms (RM) is the strongest positive predictor
- Lower status population (LSTAT) strongly negatively correlates with price
- Crime rate (CRIM) and pollution (NOX) reduce property values
- Proximity to employment centers (DIS) affects pricing

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **Scipy** - Statistical analysis
- **Jupyter Notebook** - Interactive development

## 📚 Learning Outcomes

After completing this project, you will understand:

✅ Regression analysis fundamentals  
✅ Feature correlation and multicollinearity  
✅ Regression evaluation metrics  
✅ Regularization techniques (Ridge/Lasso)  
✅ Ensemble methods for regression  
✅ Residual analysis and model diagnostics  
✅ Feature importance interpretation  
✅ Cross-validation for regression  

## 🔍 Key Regression Concepts

1. **Linear Relationships**: Understanding how features linearly relate to price
2. **Regularization**: Preventing overfitting with Ridge and Lasso
3. **Feature Selection**: Automatic feature selection with Lasso
4. **Ensemble Methods**: Combining multiple models for better predictions
5. **Residual Analysis**: Checking model assumptions and performance
6. **Cross-Validation**: Robust model evaluation techniques

## 🔄 Next Steps

1. **Feature Engineering**: Create interaction terms and polynomial features
2. **Advanced Models**: Try neural networks or XGBoost
3. **Hyperparameter Tuning**: Optimize model parameters with GridSearch
4. **Geographic Analysis**: Add location-based features
5. **Time Series**: Extend to predict price trends over time

## 📖 References

- [Boston Housing Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
- [Scikit-learn Regression Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Boston Housing - Original Paper](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

---

**🎯 Perfect for**: Learning regression fundamentals, understanding feature importance

**⏱️ Estimated Time**: 2-3 hours

**🎓 Difficulty**: Beginner-friendly with comprehensive explanations

**💡 Key Learning**: Regression metrics, feature relationships, model comparison
