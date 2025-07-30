"""
Fish weight prediction RESTful API
Production ready FastAPI service for fish weight predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fish Weight Prediction API",
    description="Production API for predicting fish weight from physical measurements",
    version="1.0.0"
)

# Fish species mapping (from your existing fish analysis)
FISH_SPECIES = {
    'Bream': 0, 'Roach': 1, 'Whitefish': 2, 'Parkki': 3, 
    'Perch': 4, 'Pike': 5, 'Smelt': 6
}

class FishMeasurements(BaseModel):
    """Fish measurement input model"""
    species: str
    length1: float  # Vertical length in cm
    length2: float  # Diagonal length in cm  
    length3: float  # Cross length in cm
    height: float   # Height in cm
    width: float    # Diagonal width in cm
    
    @validator('species')
    def validate_species(cls, v):
        if v not in FISH_SPECIES:
            raise ValueError(f'Species must be one of: {list(FISH_SPECIES.keys())}')
        return v
    
    @validator('length1', 'length2', 'length3', 'height', 'width')
    def validate_measurements(cls, v):
        if v <= 0:
            raise ValueError('Measurements must be positive')
        if v > 100:  # Reasonable max for fish measurements in cm
            raise ValueError('Measurement seems unrealistic (>100cm)')
        return v

class FishWeightPrediction(BaseModel):
    """Fish weight prediction output model"""
    predicted_weight: float
    confidence_score: float
    model_used: str
    species: str
    prediction_id: str
    timestamp: datetime
    
class BatchFishPrediction(BaseModel):
    """Batch fish prediction input"""
    fish_measurements: List[FishMeasurements]

class FishModelInfo(BaseModel):
    """Fish model information"""
    model_name: str
    accuracy_metrics: dict
    training_date: str
    fish_species_supported: List[str]

class FishWeightPredictor:
    """Fish weight prediction service"""
    
    def __init__(self, model_path: str = "/models/fish_weight_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.poly_features = None
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained fish weight prediction model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.feature_selector = model_data.get('feature_selector')
                self.poly_features = model_data.get('poly_features')
                self.model_info = model_data.get('metadata', {})
                logger.info(f"Fish weight model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Fish model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading fish model: {e}")
            raise
    
    def preprocess_fish_data(self, measurements: FishMeasurements) -> np.ndarray:
        """Preprocess fish measurements for prediction"""
        # Create feature vector with species encoding
        species_encoded = FISH_SPECIES[measurements.species]
        
        # Fish features: [Length1, Length2, Length3, Height, Width, Species]
        features = np.array([[
            measurements.length1,
            measurements.length2, 
            measurements.length3,
            measurements.height,
            measurements.width,
            species_encoded
        ]])
        
        # Apply preprocessing pipeline if available
        if self.poly_features:
            features = self.poly_features.transform(features)
        
        if self.scaler:
            features = self.scaler.transform(features)
            
        if self.feature_selector:
            features = self.feature_selector.transform(features)
            
        return features
    
    def predict_fish_weight(self, measurements: FishMeasurements) -> FishWeightPrediction:
        """Predict fish weight from measurements"""
        if not self.model:
            raise HTTPException(status_code=503, detail="Fish prediction model not available")
        
        try:
            # Preprocess measurements
            features = self.preprocess_fish_data(measurements)
            
            # Make prediction
            predicted_weight = self.model.predict(features)[0]
            
            # Calculate confidence (simplified - could use prediction intervals)
            confidence = min(0.95, max(0.7, self.model_info.get('r2_score', 0.8)))
            
            # Generate prediction ID
            prediction_id = f"fish_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return FishWeightPrediction(
                predicted_weight=round(predicted_weight, 2),
                confidence_score=round(confidence, 3),
                model_used=self.model_info.get('best_model', 'RandomForest'),
                species=measurements.species,
                prediction_id=prediction_id,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Fish weight prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize fish weight predictor
fish_predictor = FishWeightPredictor()

@app.get("/")
async def root():
    """Fish Weight Prediction API root endpoint"""
    return {
        "message": "Fish Weight Prediction API",
        "version": "1.0.0",
        "description": "Predict fish weight from physical measurements",
        "endpoints": {
            "predict": "/predict/fish-weight",
            "batch": "/predict/fish-weight/batch", 
            "health": "/health",
            "model-info": "/model/info"
        }
    }

@app.post("/predict/fish-weight", response_model=FishWeightPrediction)
async def predict_fish_weight(measurements: FishMeasurements):
    """
    Predict fish weight from physical measurements
    
    - **species**: Fish species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)
    - **length1**: Vertical length in cm
    - **length2**: Diagonal length in cm
    - **length3**: Cross length in cm  
    - **height**: Height in cm
    - **width**: Diagonal width in cm
    
    Returns predicted weight in grams with confidence score.
    """
    return fish_predictor.predict_fish_weight(measurements)

@app.post("/predict/fish-weight/batch", response_model=List[FishWeightPrediction])
async def predict_fish_weight_batch(batch_request: BatchFishPrediction):
    """
    Batch fish weight predictions for multiple fish
    
    Useful for processing multiple fish measurements at once.
    """
    predictions = []
    for measurements in batch_request.fish_measurements:
        prediction = fish_predictor.predict_fish_weight(measurements)
        predictions.append(prediction)
    
    return predictions

@app.get("/health")
async def health_check():
    """Health check endpoint for fish prediction service"""
    model_status = "loaded" if fish_predictor.model else "not_loaded"
    
    return {
        "status": "healthy" if fish_predictor.model else "degraded",
        "model_status": model_status,
        "supported_species": list(FISH_SPECIES.keys()),
        "timestamp": datetime.now()
    }

@app.get("/model/info", response_model=FishModelInfo)
async def get_fish_model_info():
    """Get information about the fish weight prediction model"""
    if not fish_predictor.model:
        raise HTTPException(status_code=503, detail="Fish model not loaded")
    
    return FishModelInfo(
        model_name=fish_predictor.model_info.get('best_model', 'Unknown'),
        accuracy_metrics={
            'r2_score': fish_predictor.model_info.get('r2_score', 0.0),
            'rmse': fish_predictor.model_info.get('rmse', 0.0),
            'mae': fish_predictor.model_info.get('mae', 0.0),
            'mape': fish_predictor.model_info.get('mape', 0.0)
        },
        training_date=fish_predictor.model_info.get('training_date', 'Unknown'),
        fish_species_supported=list(FISH_SPECIES.keys())
    )

@app.get("/species")
async def get_supported_species():
    """Get list of supported fish species"""
    return {
        "supported_species": list(FISH_SPECIES.keys()),
        "species_count": len(FISH_SPECIES),
        "encoding": FISH_SPECIES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
