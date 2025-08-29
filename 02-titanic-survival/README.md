# 🚢 Titanic Survival Predictor

**Level**: 🟢 Beginner  
**Type**: Binary Classification  
**Dataset**: Titanic Dataset (Kaggle/Seaborn built-in)

## 📋 Project Overview

This project predicts passenger survival on the RMS Titanic using machine learning. It's an excellent beginner project that introduces feature engineering, handling missing data, and working with categorical variables. The sinking of the Titanic is one of the most infamous shipwrecks in history, and this dataset provides rich information about passenger demographics and survival outcomes.

## 🎯 Objectives

- Learn advanced data preprocessing techniques
- Handle missing values effectively
- Perform feature engineering and creation
- Work with categorical and numerical data
- Build and compare multiple classification models
- Understand the importance of feature selection

## 📊 Dataset Information

The Titanic dataset contains information about passengers aboard the RMS Titanic.

### Features (Input Variables)
- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

### Target Variable
- **Survived**: Survival status (0 = No, 1 = Yes)

### Dataset Statistics
- **Total Samples**: 891 passengers
- **Features**: 11 features (mix of numerical and categorical)
- **Missing Values**: Age (~20%), Cabin (~77%), Embarked (~0.2%)
- **Class Distribution**: ~38% survived, ~62% did not survive

## 🔍 Project Structure

```
02-titanic-survival/
├── notebook.ipynb          # Main Jupyter notebook
├── README.md              # This file
├── requirements.txt       # Project dependencies
├── data/                  # Dataset files
│   └── titanic.csv       # Titanic dataset
└── results/              # Generated plots and results
    ├── survival_by_class.png
    ├── age_distribution.png
    ├── correlation_matrix.png
    └── feature_importance.png
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
cd 02-titanic-survival
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
- Load the Titanic dataset
- Explore dataset structure and basic statistics
- Identify missing values and data types
- Analyze survival rates by different factors

### 2. Exploratory Data Analysis (EDA)
- Survival rate analysis by passenger class, gender, age
- Fare distribution and its relationship with survival
- Family size impact on survival
- Port of embarkation analysis
- Correlation analysis between features

### 3. Data Preprocessing & Feature Engineering
- **Missing Value Treatment**:
  - Age: Imputation based on title and class
  - Embarked: Mode imputation
  - Cabin: Create binary feature for cabin availability
- **Feature Creation**:
  - Extract titles from names (Mr, Mrs, Miss, etc.)
  - Create family size feature (SibSp + Parch + 1)
  - Create fare bins for better categorization
  - Age groups creation
- **Encoding**: Convert categorical variables to numerical

### 4. Model Building & Training
- **Logistic Regression**: Baseline linear model
- **Decision Tree**: Non-linear, interpretable model
- **Random Forest**: Ensemble method for better accuracy
- **Gradient Boosting**: Advanced ensemble technique
- **Support Vector Machine**: Margin-based classifier

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Confusion Matrix
- Cross-validation scores
- Feature importance analysis

### 6. Model Optimization
- Hyperparameter tuning using GridSearchCV
- Feature selection techniques
- Model ensemble methods

## 📊 Expected Results

### Model Performance (Expected)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~81% | ~79% | ~74% | ~76% | ~85% |
| Decision Tree | ~78% | ~76% | ~71% | ~73% | ~82% |
| Random Forest | ~83% | ~81% | ~76% | ~78% | ~87% |
| Gradient Boosting | ~84% | ~82% | ~77% | ~79% | ~88% |
| SVM | ~82% | ~80% | ~75% | ~77% | ~86% |

### Key Insights
- Gender is the strongest predictor (women had higher survival rates)
- Passenger class significantly affects survival chances
- Age and family size also play important roles
- Fare amount correlates with survival probability

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Jupyter Notebook** - Interactive development

## 📚 Learning Outcomes

After completing this project, you will understand:

✅ Advanced data preprocessing techniques  
✅ Handling missing values strategically  
✅ Feature engineering and creation  
✅ Working with categorical variables  
✅ Model comparison and selection  
✅ Hyperparameter tuning  
✅ Feature importance analysis  
✅ Business insights from data  

## 🔍 Key Features Engineered

1. **Title Extraction**: Extract titles from names (Mr, Mrs, Miss, Master, etc.)
2. **Family Size**: Combine SibSp and Parch to create family size
3. **Is Alone**: Binary feature indicating if passenger traveled alone
4. **Age Groups**: Categorize ages into meaningful groups
5. **Fare Bins**: Create fare categories for better analysis
6. **Cabin Available**: Binary feature for cabin information availability

## 🔄 Next Steps

1. **Advanced Feature Engineering**: Create interaction features
2. **Deep Learning**: Try neural networks for comparison
3. **Ensemble Methods**: Combine multiple models
4. **Web Deployment**: Create a web app for predictions
5. **Real-time Predictions**: Build an API for the model

## 📖 References

- [Titanic Dataset - Kaggle](https://www.kaggle.com/c/titanic)
- [RMS Titanic - Wikipedia](https://en.wikipedia.org/wiki/RMS_Titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**🎯 Perfect for**: Learning feature engineering, handling real-world messy data

**⏱️ Estimated Time**: 3-4 hours

**🎓 Difficulty**: Beginner with intermediate concepts
