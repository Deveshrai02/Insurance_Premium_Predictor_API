from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from typing import  Any
from model.predict import  model, MODEL_VERSION  , predict_output
import pandas as pd
import logging

logger = logging.getLogger("insurance_premium_predictor")

app = FastAPI()


@app.get('/')
def read_root():
    return {"message": "Welcome to the Insurance Premium Predictor API. Use the /predict endpoint to get predictions."}

@app.get('/health')
def health_check():
    if model is not None:
        return JSONResponse(status_code=200, content={'status': 'ok', 'model_loaded': True, 'model_version': MODEL_VERSION})
    else:
        return JSONResponse(status_code=503, content={
            'status': 'error',
            'model_loaded': False,
            'message': 'Model not loaded. Ensure model file exists and is accessible.'
        })

@app.post('/predict' , response_model=PredictionResponse)
def predict_premium(data: UserInput):

    # Ensure model is loaded; if not, return a helpful error instead of crashing
    if model is None:
        return JSONResponse(status_code=500, content={
            'error': 'Model not loaded. Ensure model file exists and is accessible.'
        })

    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
        prediction = predict_output(user_input)
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={
            'error': 'Prediction failed',
            'detail': str(e)
        })

    return JSONResponse(status_code=200, content={'response': prediction})




