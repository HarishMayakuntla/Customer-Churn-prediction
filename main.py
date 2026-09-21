from fastapi import FastAPI
import pandas as pd
import joblib

app=FastAPI(
    title='customer churn prediction API',
    description='API for churn customers',
    version='1.0.0'
)

model=joblib.load('churn_prediction_model.pkl')


@app.get('/')
def home():
    return{'meassage':'customer churn prediction API running'}


@app.get('/predict')
def predict(
    Gender: str,
    Contract: str,
    PaymentMethod: str,
    Age: int,
    Tenure: int,
    MonthlyCharges: float,
    TotalCharges: float
):

    data=pd.DataFrame([{
            "Gender": Gender,
            "Contract": Contract,
            "PaymentMethod": PaymentMethod,
            "Age": Age,
            "Tenure": Tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
    }])


    proba=model.predict_proba(data)[0][1]

    predict=model.predict(data)[0]

    if proba >=0.34:  # thresold 0.34
        predict=1
        churn='yes'

    else :
        predict=0
        churn='no'


    return{
        'prediction':int(predict),
        'churn':churn,
        'probabulity':round(float(proba),4)

    }

