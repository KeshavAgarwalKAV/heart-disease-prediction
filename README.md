# 🫀 Heart Disease Prediction — End-to-End ML Classification

A complete machine learning pipeline to predict the presence of heart disease in patients using clinical data, built with Python and scikit-learn.

---

## 📌 Problem Statement

> Given clinical parameters about a patient (age, cholesterol, chest pain type, etc.), can we predict whether or not they have heart disease?

This is a binary classification problem with real-world clinical significance — early detection of heart disease can be life-saving.

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Size:** 303 patients, 14 clinical features
- **Target:** `1` = Heart disease present, `0` = No heart disease

---

## 📖 Data Dictionary

- age: Age in years  
- sex: 1 = male, 0 = female  
- cp: Chest pain type (0–3)  
- trestbps: Resting blood pressure (>130–140 risky)  
- chol: Serum cholesterol (>200 risky)  
- fbs: Fasting blood sugar >120 mg/dl  
- restecg: ECG results  
- thalach: Maximum heart rate achieved  
- exang: Exercise-induced angina  
- oldpeak: ST depression  
- slope: ST segment slope  
- ca: Number of major vessels (0–3)  
- thal: Thalium stress test  
- target: Heart disease (1 = yes, 0 = no)

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Target class distribution analysis
- Correlation heatmap across all 14 features
- Cross-tabulations: heart disease vs sex, chest pain type
- Scatter plots: age vs max heart rate coloured by disease status

### 2. Model Comparison
Three models were trained and compared before tuning:

| Model | Baseline Accuracy |
|-------|-------------------|
| Logistic Regression | ~88.5% |
| K-Nearest Neighbors | ~68.9% |
| Random Forest | ~83.6% |

### 3. Hyperparameter Tuning
- **KNN:** Tuned `n_neighbors` (1–20) by hand, comparing train vs test scores
- **Logistic Regression:** Tuned using `RandomizedSearchCV` then `GridSearchCV`
- **Random Forest:** Tuned `max_depth`, `max_features`, `min_samples_leaf`, `n_estimators`

### 4. Best Model: Tuned Logistic Regression
Cross-validated results (5-fold CV):

| Metric | Score |
|--------|-------|
| **Accuracy** | **88.5%** |
| Precision | 86.6% |
| Recall | 92.1% |
| F1-Score | 89.2% |

### 5. Feature Importance
Top predictors identified via logistic regression coefficients:
- `cp` (chest pain type) — strongest positive predictor
- `slope` — positive correlation with disease
- `ca` — negative correlation (more vessels = less disease)
- `thal` — strong diagnostic indicator

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, Matplotlib, Seaborn
- **scikit-learn** — LogisticRegression, KNeighborsClassifier, RandomForestClassifier
- **Model Selection** — GridSearchCV, RandomizedSearchCV, cross_val_score

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/KeshavAgarwalKAV/heart-disease-prediction
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook end_to_end_heart_disease_classification.ipynb
```

**requirements.txt:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── end_to_end_heart_disease_classification.ipynb  # Main notebook
├── heart-disease.csv                               # Dataset
└── README.md
```

---

## 📈 Results Summary

- Achieved **88.5% cross-validated accuracy** with tuned Logistic Regression
- **Recall of 92.1%** — critical in medical diagnosis (minimizing false negatives)
- Identified `cp`, `slope`, `thal`, and `ca` as the most clinically significant features

---

## 🔮 Future Improvements

- Deploy as a Streamlit web app for real-time predictions
- Try ensemble methods (XGBoost, LightGBM) for potential accuracy gains
- Incorporate SHAP values for better model explainability
