from fastapi import FastAPI,  HTTPException
from pydantic import BaseModel  
import uvicorn
import pickle


# Define a Pydantic model for the request body
class DementiaInput(BaseModel):
    diabetic: int
    alcohollevel: int
    heartrate: int
    bloodoxygenlevel: int
    bodytemperature: int
    weight: int
    mri_delay: int
    age: int
    education_level: int
    dominant_hand: int
    gender: int
    family_history: int
    smoking_status: int
    physical_activity: int
    depression_status: int
    cognitive_test_scores: int
    medication_history: int
    nutrition_diet: int
    sleep_quality: int

with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
async def root():
    return {'Text': 'Dementia Predictions'}

@app.get("/predict/")
async def predict_dementia(input_data: DementiaInput):
    # Extract features from input_data and make prediction using your model
    features = [input_data.diabetic, input_data.alcohollevel, input_data.heartrate, input_data.bloodoxygenlevel,
                input_data.bodytemperature, input_data.weight, input_data.mri_delay, input_data.age,
                input_data.cognitive_test_scores]
    prediction = model.predict([features])[0]
    
    # Convert prediction to human-readable format
    prediction_label = "Dementia" if prediction == 1 else "No Dementia"
    
    # Return prediction result
    return {"prediction": prediction_label}





