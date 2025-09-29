# Telco Customer Churn Prediction with Explainable AI

**An Explainable AI solution to predict customer churn in subscription-based businesses using machine learning models, and to deliver interpretable insights via SHAP & LIME.**

---

## 1. Project Overview  
Customer churn is a critical risk for subscription-based businesses (e.g. telecoms). Acquiring new customers is typically more expensive than retaining existing ones.  
This project develops predictive models to identify customers likely to churn and explains **why** these predictions are made using Explainable AI techniques. This combination of **prediction + interpretation** empowers businesses to make better data-driven decisions and reduce churn effectively.

---

## 2. Key Highlights  
- Implemented **five ML models**: XGBoost, LightGBM, Random Forest, AdaBoost, Naïve Bayes.  
- Compared performance using **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.  
- Achieved **highest accuracy (0.78)** with LightGBM and **best ROC-AUC (0.85)** with XGBoost.  
- Applied **SMOTE** to address class imbalance.  
- Integrated **SHAP** (global interpretability) and **LIME** (local interpretability) to make models explainable.  
- Identified **top churn drivers** such as contract type, tenure, monthly charges, and support services.  

---

## 3. Machine Learning Models & Results  

| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Comments |
|---------------|----------|-----------|--------|----------|---------|----------|
| **XGBoost**   | 0.76     | 0.54      | **0.70** | 0.61     | **0.85** | Best ROC-AUC, strong balance between sensitivity & specificity |
| **LightGBM**  | **0.78** | 0.57      | 0.63   | 0.60     | 0.84    | Highest accuracy, efficient and scalable |
| Random Forest | 0.77     | 0.56      | 0.57   | 0.57     | 0.83    | Stable performance, interpretable feature importance |
| AdaBoost      | 0.70     | 0.47      | **0.90** | 0.62     | 0.84    | Excellent recall (captures most churners), but lower precision |
| Naïve Bayes   | 0.72     | 0.49      | 0.77   | 0.60     | 0.82    | Simple baseline, robust but less accurate |

**Key Takeaway**:  
- **LightGBM** achieved the best overall accuracy.  
- **XGBoost** had the strongest ROC-AUC (0.85), making it the most reliable classifier for distinguishing churn vs. non-churn customers.  
- **AdaBoost** excelled in Recall (0.90), meaning it rarely misses churners — though at the cost of more false positives.  

---

## 4. Tech Stack  
- **Language**: Python 3.10
- **Notebook / IDE**: Google Colab  
#### Core Libraries
- **Data Processing & Visualization**:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Machine Learning & Preprocessing**:  
  - `scikit-learn` (LabelEncoder, MinMaxScaler, StandardScaler, feature selection, model selection, metrics)  
  - `RandomForestClassifier`, `AdaBoostClassifier`, `GaussianNB`  
- **Boosting Algorithms**:  
  - `xgboost` (XGBClassifier)  
  - `lightgbm` (LGBMClassifier)  
- **Class Imbalance Handling**:  
  - `imbalanced-learn` (`SMOTE`, `RandomUnderSampler`, `Pipeline`)  
- **Explainable AI (XAI)**:  
  - `shap`, `lime`  

#### Utilities
- `warnings` — suppress irrelevant warnings  
- `collections.Counter` — handle class distributions
---

## 5. Exploratory Data Analysis (EDA)  
The Telco Customer Churn dataset (7,043 customers, 21 attributes) was explored to uncover patterns:  
- **Churn distribution**: Imbalanced (approx. 26% churners, 74% non-churners).  
- **Categorical features**: Strong churn correlation with contract type, tech support, and payment method.  
- **Numerical features**: Shorter tenure and higher monthly charges linked with higher churn.  
- **Correlation analysis**: Heatmaps revealed moderate correlations (e.g., tenure negatively correlated with churn).  

---

## 6. Model Training & Evaluation  
- **Preprocessing**:  
  - Cleaned missing/invalid values  
  - Encoded categorical variables (Label Encoding, One-Hot)  
  - Normalized numeric features  
- **Balancing**: Applied **SMOTE** to address class imbalance  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, ROC-AUC  
- **Cross-validation**: Ensured generalization on unseen data  
- **Explainability Layer**:  
  - **SHAP** → ranked global feature importance (e.g., tenure, contract, monthly charges)  
  - **LIME** → provided customer-level churn explanations  

---

## 7. Insights & Recommendations  
**Insights from XAI**:  
- Customers on **month-to-month contracts** were at the highest churn risk.  
- **Short tenure** strongly increased churn likelihood.  
- **High monthly charges** without additional benefits drove churn.  
- **Lack of tech support** was a major risk factor.  

**Business Recommendations**:  
- Offer **loyalty discounts** or incentives for customers at early tenure stages.  
- Promote **longer-term contracts** with added value.  
- Bundle **tech support & online services** to reduce risk.  
- Use **LIME explanations** in CRM systems to prioritize high-risk customers with actionable retention strategies.  

---

## 8. Repository Structure  
```text
telco_customer_churn/
│
├── data/
│   └── Telco-Customer-Churn.csv  
│
├── notebooks/
│   └── CP2_Source_Code.ipynb  
│
├── requirements.txt   
└── README.md 
