from fastapi import FastAPI
from pydantic import BaseModel
import joblib # ou import pickle
import numpy as np

app = FastAPI()

# Chargement sécurisé du modèle
try:
    model = joblib.load("random_forest_model.pkl")
    print("✅ Modèle chargé !")
except:
    model = None
    print("❌ Modèle introuvable !")

class PatientData(BaseModel):
    age: int
    gender: int
    height: int
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

@app.post("/predict")
async def predict(data: PatientData):
    if model is None: return {"error": "Modèle non chargé"}
    
    features = np.array([[data.age, data.gender, data.height, data.weight, 
                          data.ap_hi, data.ap_lo, data.cholesterol, data.gluc, 
                          data.smoke, data.alco, data.active]])
    
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    
    return {"prediction": prediction, "probability": probability}