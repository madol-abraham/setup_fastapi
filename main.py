##main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import os
import logging
from typing import Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mock model - replace with your actual model loading code
class PredictionModel:
    def predict(self, input_data: np.ndarray) -> float:
        # Replace with actual model prediction
        return np.random.random()  # Mock prediction

model = PredictionModel()

# Pydantic models for request validation
class PredictionInput(BaseModel):
    Soil_Moisture: float
    temperature: float
    Time: float
    Wind_speed_km_h: float
    Air_humidity_percent: float
    rainfall: float
    Soil_Type: int
    Crop_Type: int

class RetrainInput(BaseModel):
    epochs: int = 10
    batch_size: int = 32

@app.post("/api/predict")
async def predict(input_data: PredictionInput):
    """Make irrigation prediction"""
    try:
        features = np.array([[
            input_data.Soil_Moisture,
            input_data.temperature,
            input_data.Time,
            input_data.Wind_speed_km_h,
            input_data.Air_humidity_percent,
            input_data.rainfall,
            input_data.Soil_Type,
            input_data.Crop_Type
        ]], dtype=np.float32)

        prediction = model.predict(features)
        predicted_class = int(prediction >= 0.5)
        confidence = round(prediction if predicted_class else 1 - prediction, 4)

        return {
            "irrigation_needed": predicted_class,
            "confidence": confidence,
            "raw_prediction": float(prediction)
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/api/retrain")
async def retrain_model(
    file: UploadFile = File(...),
    epochs: int = 10,
    batch_size: int = 32
):
    """Retrain model with new data"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, "Only CSV files accepted")

        # Save uploaded file (in production, use proper storage)
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Here you would add your actual retraining logic
        # For now just mock the response
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "accuracy": 0.95,  # Mock accuracy
            "file": file.filename
        }
    
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        raise HTTPException(500, f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)