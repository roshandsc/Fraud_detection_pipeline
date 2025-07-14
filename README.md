# Fraud Detection Pipeline

A **scalable machine learning pipeline** for detecting fraudulent transactions using advanced feature engineering and ensemble models.

---

##  Overview

This project is an **end-to-end fraud detection pipeline**, built to detect fraudulent transactions in large-scale financial data.  
The system combines:

- Classical supervised machine learning approaches
- Hybrid ensemble techniques
- Unsupervised anomaly detection methods

It is designed to ensure **high accuracy and robustness**, even in challenging, real-world scenarios.  
We used the IEEE-CIS Fraud Detection dataset, addressing issues such as data imbalance, outliers, and high-dimensional features.

---

##  Objectives

- Build a complete fraud detection pipeline, from raw data to deployment-ready models.
- Compare multiple supervised learning algorithms (Logistic Regression, Decision Trees, Random Forest, Boosting).
- Explore hybrid models and unsupervised anomaly detection approaches.
- Handle imbalanced datasets effectively using oversampling (e.g., SMOTE) and class weighting.
- Develop a **modular, reusable, and scalable code structure**.

---
## Pipeline Architecture 

Raw Data → Preprocessing → Feature Engineering → Handling Imbalance → Model Training → Hybrid / Unsupervised Detection → Evaluation → Deployment-ready Pipeline

## 📚 Table of Contents

- [Basic Terms & Definitions](#basic-terms--definitions)
- [Steps from EDA to Pipeline](#steps-from-eda-to-pipeline)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering & Selection](#3-feature-engineering--selection)
  - [4. Model Selection & Training](#4-model-selection--training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Pipeline Automation & Deployment](#6-pipeline-automation--deployment)
- [Detailed Code Explanation](#detailed-code-explanation)
- [Logical Conclusion & Learnings](#logical-conclusion--learnings)

---

## Basic Terms & Definitions

### Exploratory Data Analysis (EDA)

A crucial first step to understand data distribution, identify patterns, correlations, and outliers. Helps shape preprocessing and feature engineering strategies.

### Data Cleaning

The process of handling missing values, outliers, and inconsistent data entries to prepare a robust dataset.

### Feature Engineering

Creating new variables or transforming existing ones to improve model performance. E.g., ratios, interaction features, encoding, scaling.

### Model Training

The process where a machine learning algorithm learns from historical data to make predictions.

### Model Evaluation

Assessing the model's performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### Pipeline Automation

Structuring preprocessing, training, and prediction steps into a single, reusable pipeline for scalability and easier deployment.

### Deployment

Making the final model available as a service or integrated into a production system to perform live predictions.

---

## Steps from EDA to Pipeline

### 1. Exploratory Data Analysis (EDA)

- Examine class imbalance.
- Check feature distributions (histograms, KDE plots).
- Analyze correlations using heatmaps.

### 2. Data Cleaning

- Handle missing values via imputation or removal.
- Detect and remove outliers using Z-score or IQR methods.

### 3. Feature Engineering & Selection

- Create new features (e.g., transaction amount ratios).
- Encode categorical variables (e.g., one-hot, target encoding).
- Scale numerical features using StandardScaler or MinMaxScaler.
- Optionally, apply dimensionality reduction (e.g., PCA).

### 4. Model Selection & Training

- Start with baseline models (Logistic Regression, Decision Trees).
- Move to advanced models (Random Forest, XGBoost, LightGBM, CatBoost).
- Handle imbalanced data using SMOTE or class weighting.

### 5. Model Evaluation

- Use confusion matrix, classification report, and ROC curves.
- Compare precision, recall, and F1-score.
- Use cross-validation to assess generalization.

### 6. Pipeline Automation & Deployment

- Combine preprocessing and model into a `Pipeline` object.
- Serialize model using joblib or pickle.
- Integrate into an API or batch scoring pipeline.

---

## Detailed Code Explanation


### 1. **Setup and Imports**

#### Core Libraries

```python
import numpy as np
import pandas as pd
```
	•	NumPy: Fundamental package for numerical computations.
	•	Pandas: Data manipulation and analysis using DataFrames.

#### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import shap
```
	•	Matplotlib & Seaborn: For plotting distributions, correlations, and evaluation metrics.
	•	SHAP: For interpreting and explaining model predictions (feature importance, SHAP values).

#### Preprocessing & Feature Engineering

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
```
	•	StandardScaler: Scales features to mean 0 and variance 1.
	•	LabelEncoder: Encodes categorical labels numerically.
	•	PCA: Reduces dimensionality to simplify high-dimensional data.
	•	Pipeline: Combines preprocessing and modeling into one seamless workflow.
	•	SelectFromModel & mutual_info_classif: For feature selection and ranking feature importance.

 ####  Model Selection & Training
 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
```
	•	Classical Models: Logistic Regression, Decision Trees, KNN.
	•	Ensemble Models: Random Forest, Bagging, Stacking, Gradient Boosting — help improve performance and reduce overfitting.
	•	Unsupervised Models: One-Class SVM, Isolation Forest, KMeans, DBSCAN — for anomaly detection and clustering fraudulent behavior.

#### Imbalanced Data Handling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
```
	•	SMOTE: Over-samples minority class to address imbalance.
	• RandomUnderSampler: Reduces majority class to balance the dataset.

#### Advanced Boosting Libraries

```python
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
```
	•	XGBoost, LightGBM, CatBoost: Advanced gradient boosting frameworks known for excellent performance on structured data.

#### Utilities

```python
import pickle
import time
from pympler import asizeof
import warnings
warnings.filterwarnings('ignore')
```
	•	pickle: Serialize and save models for reuse or deployment.
	•	time: Track execution time for training or predictions.
	•	pympler.asizeof: Check memory usage of data or models.
	•	warnings.filterwarnings(‘ignore’): Suppresses warnings to keep notebook output clean (use with caution).
---

### 2. **Load Data**

Below, we load the transaction and identity datasets and merge them to create a unified DataFrame for analysis.

```python
train_identity = pd.read_csv("/Users/roshanshetty/Downloads/PROJECT_PHASE_1/dataset/train_identity.csv")
train_transaction = pd.read_csv("/Users/roshanshetty/Downloads/PROJECT_PHASE_1/dataset/train_transaction.csv")

# Merge datasets on TransactionID
df = pd.merge(train_identity, train_transaction, on='TransactionID', how='left')
```
### 3. **EDA and Preprocessing**

#### **Dataset Introduction**

After merging, we inspect the first few rows and data structure to understand the dataset shape and types.

```python
print(df.head())
print(df.info())
```
   | TransactionID | id_01 | id_02  | id_03 | id_05 | ....... | V331 |
|---------------|-------|--------|-------|-------|------|------|
| 2987004       | 0.0   | 70787  | NaN   | NaN  |....... |0.0  | 0.0 |
| 2987008       | -5.0  | 98945  | NaN   | 0.0  |....... | 0.0  | 0.0 |
| 2987010       | -5.0  |191631  | 0.0  | 0.0  |........ | NaN | NaN |
| 2987011       | -5.0  |221832  | NaN  | 0.0  |........ |NaN | NaN |
| 2987016       | 0.0   | 7460   | 0.0  | 1.0  |........ |0.0  | 0.0 |

[5 rows x 434 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 144233 entries, 0 to 144232
Columns: 434 entries, TransactionID to V339
dtypes: float64(399), int64(4), object(31)
memory usage: 477.6+ MB
None

### 💳 Transaction Dataset

The transaction dataset records various types of financial transactions, including money transfers, ticket bookings, and purchases. It contains the following key attributes:

1. **TransactionDT**: A time-delta value representing the time elapsed since a reference point, but not an actual timestamp.
2. **TransactionAMT**: The transaction amount in USD.
3. **ProductCD**: The product code associated with the transaction.
4. **card1 – card6**: Details related to the payment card used for the transaction.
5. **addr1 and addr2**: The purchaser's address information.
6. **dist1 and dist2**: The geographical distance between two transaction locations.
7. **P_emaildomain and R_emaildomain**: The email domains of the purchaser and recipient.
8. **C1 – C14**: Counts representing how many addresses are linked to the card.
9. **D1 – D15**: Time-based deltas indicating the number of days between consecutive transactions.
10. **M1 – M9**: Matching indicators, such as whether the cardholder's name matches the provided address.
11. **V1 – V339**: Entity relationship attributes provided by Vesta, capturing additional transaction details.

---

### 🏷️ About V Features (V1 – V339)

**VXXX** columns represent information provided by **Vesta**, specifying various entity relationships.

#### Vesta-Engineered Features

Vesta is a fraud detection and payment processing company specializing in machine learning models for transaction risk assessment.  
The Vesta-engineered features (V1 to V339) are derived from raw transaction data to enhance fraud detection.

---

#### Types of Features in V1 to V339

- **Ranking-Based Features**  
  How a transaction compares to others (e.g., is this amount unusually high for this user?).  
  *Example*: Rank of transaction amount within a certain timeframe.

- **Counting-Based Features**  
  How often a particular entity (e.g., card, email, IP address) appears.  
  *Example*: Number of transactions from the same card in the past week.

- **Entity Relationship Features**  
  Connections between users, devices, locations, and payment methods.  
  *Example*: Is this IP address linked to multiple different cards?

- **Time-Series Features**  
  Trends in transaction behavior over time.  
  *Example*: Time gap between consecutive transactions for a user.

---

#### Why Are These Features Important?

- Detect unusual patterns that indicate fraud.
- Enhance model accuracy by encoding risk factors from raw transaction data.
- Help prevent fraud in real-time by identifying suspicious behaviors.

---
####  Exploratory Data Analysis (EDA)

A crucial first step in fraud detection is understanding the **class imbalance**, as fraudulent transactions are usually much rarer than genuine ones.  
Here, we visualize the distribution of fraud vs. non-fraud cases using bar and pie charts.

```python
# Bar plot to show class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=df)
plt.title('Class Distribution (Bar Plot)')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()

# Pie chart to show class proportion
plt.figure(figsize=(6, 6))
df['isFraud'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
plt.title('Class Distribution (Pie Chart)')
plt.ylabel('')
plt.show()
```
	•	The bar plot clearly shows the imbalance between fraudulent and non-fraudulent transactions, with fraud cases forming a very small fraction.
	•	The pie chart provides a proportional view, helping understand the severity of class imbalance visually.
 
