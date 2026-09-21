# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to **churn (Yes/No)** based on customer demographic, contract, payment, and billing information.

The trained model is integrated with **FastAPI** to provide real-time customer churn predictions through a REST API.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services.

In this project, customer data is processed and used to train a **Gradient Boosting Classifier**. Hyperparameter tuning and threshold tuning were performed to improve the classification performance.

The final trained model is saved as a `.pkl` file and deployed using FastAPI.

---

## 📂 Project Structure

```text
Customer-Churn-prediction/
│
├── Customer Churn Prediction .ipynb
├── customer_churn.csv
├── churn_prediction_model.pkl
├── main.py
├── requirements.txt
├── output_screenshot/
├── README.md
└── LICENSE
```

---

## 📊 Dataset

The dataset contains customer information related to demographics, contracts, payments, and charges.

### Features Used

#### Categorical Features

* `Gender`
* `Contract`
* `PaymentMethod`

#### Numerical Features

* `Age`
* `Tenure`
* `MonthlyCharges`
* `TotalCharges`

### Target

* `Churn`

  * `0` → No
  * `1` → Yes

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Gradient Boosting Classifier
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
FastAPI Deployment
```

---

## 🧹 Data Preprocessing

### Categorical Encoding

Categorical features were converted into numerical values using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

### Numerical Scaling

Numerical features were scaled using:

```python
StandardScaler()
```

A `ColumnTransformer` was used to apply the appropriate preprocessing to categorical and numerical features.

---
## 🤖 Machine Learning Models

Multiple Machine Learning classification algorithms were trained and evaluated on the dataset.

The models were compared using metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Models Evaluated

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* HistGradientBoosting Classifier
* XGBoost Classifier

After comparing the model performance, **Gradient Boosting Classifier** was selected as the final model for this project.

### Final Model

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    random_state=30
)
```

The selected Gradient Boosting model was further 
Hyperparameter tuning was performed using `GridSearchCV`,evaluated using threshold tuning.

---

## 🎯 Threshold Tuning

The default classification threshold is `0.5`.

Since customer churn is a binary classification problem, threshold tuning was performed to improve the detection of churn cases.

The selected threshold was approximately:

```text
0.34
```

The threshold was selected using validation data before evaluating the final model on the test data.

---

## 📈 Model Performance

Final test-set performance:

| Metric           |    Score |
| ---------------- | -------: |
| Accuracy         | **0.76** |
| Macro F1-Score   | **0.72** |
| Class 0 F1-Score | **0.83** |
| Class 1 F1-Score | **0.61** |

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.80      0.86      0.83     13371
           1       0.67      0.56      0.61      6629

    accuracy                           0.76     20000
   macro avg       0.73      0.71      0.72     20000
weighted avg       0.76      0.76      0.76     20000
```

---

## 💾 Model Saving

The trained model was saved using Joblib:

```python
import joblib

joblib.dump(model, "churn_prediction_model.pkl")
```

The saved model contains the trained machine learning pipeline and can be loaded for prediction.

---

# ⚡ FastAPI Deployment

The trained model is integrated with FastAPI to provide a prediction API.

### Install Dependencies

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

### Run the FastAPI Application

```bash
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

## 📚 API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

From Swagger UI, you can test the `/predict` endpoint directly.

---

# 🔮 Prediction API

### Endpoint

```text
POST /predict
```

### Sample Input

```json
{
    "Gender": "Male",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "Age": 35,
    "Tenure": 5,
    "MonthlyCharges": 70.5,
    "TotalCharges": 352.5
}
```

### Sample Response

```json
{
    "prediction": 1,
    "churn": "Yes",
    "probability": 0.6842
}
```

### Response Meaning

* `prediction: 0` → Customer is predicted **not to churn**
* `prediction: 1` → Customer is predicted **to churn**
* `probability` → Probability of the customer belonging to the churn class

> The probability shown above is an example. The actual value depends on the trained model and input data.

---

## 🖥️ API Screenshots

Screenshots of the model results and FastAPI API can be found in:

```text
output_screenshot/
```

---

## 📌 Key Learning Outcomes

Through this project, I worked with:

* Data preprocessing
* Categorical encoding
* Feature scaling
* ColumnTransformer
* Gradient Boosting
* Hyperparameter tuning
* Threshold tuning
* Classification metrics
* Confusion matrix
* ROC-AUC evaluation
* Model serialization using Joblib
* FastAPI REST API development
* Machine Learning model deployment

---

## 👨‍💻 Author

**Harish Mayakuntla**

GitHub: [HarishMayakuntla](https://github.com/HarishMayakuntla)

LinkedIn: [Harish Mayakuntla](https://www.linkedin.com/in/harish-mayakuntla/)

---

## 📄 License

This project is licensed under the MIT License.
