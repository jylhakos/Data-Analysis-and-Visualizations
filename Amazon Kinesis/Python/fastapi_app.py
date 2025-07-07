from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import boto3
import json
import logging
from pydantic import BaseModel
import uvicorn
from temperature_ml_model import TemperatureForecastModel
from kinesis_processor import KinesisDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IoT Temperature Forecasting API",
    description="API for IoT temperature sensor data processing and forecasting",
    version="1.0.0"
)

# Global variables
ml_model = TemperatureForecastModel()
kinesis_processor = None

# Pydantic models for API
class SensorReading(BaseModel):
    sensor_id: str
    temperature: float
    humidity: Optional[float] = None
    timestamp: Optional[datetime] = None
    location: Optional[str] = None

class ForecastRequest(BaseModel):
    sensor_id: str
    hours_ahead: int = 24

class ForecastResponse(BaseModel):
    sensor_id: str
    predictions: List[Dict[str, Any]]
    generated_at: datetime

class HistoricalDataRequest(BaseModel):
    sensor_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 1000

# In-memory storage for demo (replace with actual database)
sensor_data_store = []

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global kinesis_processor
    logger.info("Starting IoT Temperature Forecasting API...")
    
    # Initialize Kinesis processor
    kinesis_processor = KinesisDataProcessor("temperature-sensor-stream")
    
    # Try to load existing model
    try:
        ml_model.load_model("temperature_model.joblib")
        logger.info("Loaded existing ML model")
    except:
        logger.info("No existing model found. Will train when data is available.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IoT Temperature Forecasting API",
        "version": "1.0.0",
        "endpoints": [
            "/sensor-data",
            "/forecast",
            "/historical-data",
            "/model/train",
            "/model/status"
        ]
    }

@app.post("/sensor-data")
async def receive_sensor_data(reading: SensorReading):
    """Receive new sensor data"""
    try:
        # Add timestamp if not provided
        if reading.timestamp is None:
            reading.timestamp = datetime.utcnow()
        
        # Store data (in production, this would go to a database)
        sensor_data_store.append(reading.dict())
        
        # Send to Kinesis (optional, if you want to also send via API)
        await send_to_kinesis(reading)
        
        logger.info(f"Received data from sensor {reading.sensor_id}: {reading.temperature}°C")
        
        return {
            "status": "success",
            "message": "Sensor data received",
            "sensor_id": reading.sensor_id,
            "timestamp": reading.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error receiving sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def get_temperature_forecast(request: ForecastRequest) -> ForecastResponse:
    """Get temperature forecast for a specific sensor"""
    try:
        if not ml_model.is_trained:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first."
            )
        
        # Get recent data for the sensor
        sensor_data = [
            data for data in sensor_data_store 
            if data['sensor_id'] == request.sensor_id
        ]
        
        if not sensor_data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for sensor {request.sensor_id}"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(sensor_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Get recent data (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_data = df[df['timestamp'] >= recent_cutoff]
        
        if recent_data.empty:
            raise HTTPException(
                status_code=400,
                detail="No recent data available for prediction"
            )
        
        # Make predictions
        predictions_df = ml_model.predict_next_24h(recent_data)
        
        # Format response
        predictions = predictions_df.to_dict('records')
        for pred in predictions:
            pred['timestamp'] = pred['timestamp'].isoformat()
        
        return ForecastResponse(
            sensor_id=request.sensor_id,
            predictions=predictions,
            generated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical-data")
async def get_historical_data(
    sensor_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 1000
):
    """Get historical sensor data"""
    try:
        # Filter data
        filtered_data = sensor_data_store.copy()
        
        if sensor_id:
            filtered_data = [d for d in filtered_data if d['sensor_id'] == sensor_id]
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            filtered_data = [
                d for d in filtered_data 
                if datetime.fromisoformat(d['timestamp']) >= start_dt
            ]
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            filtered_data = [
                d for d in filtered_data 
                if datetime.fromisoformat(d['timestamp']) <= end_dt
            ]
        
        # Apply limit
        filtered_data = filtered_data[-limit:]
        
        return {
            "data": filtered_data,
            "count": len(filtered_data),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train the ML model with available data"""
    try:
        if len(sensor_data_store) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for training. Need at least 100 data points."
            )
        
        # Train model in background
        background_tasks.add_task(train_model_background)
        
        return {
            "status": "success",
            "message": "Model training started in background",
            "data_points": len(sensor_data_store)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get model training status and metrics"""
    return {
        "is_trained": ml_model.is_trained,
        "available_data_points": len(sensor_data_store),
        "last_updated": datetime.utcnow().isoformat()
    }

@app.get("/sensors")
async def list_sensors():
    """Get list of all sensors that have sent data"""
    sensor_ids = list(set(data['sensor_id'] for data in sensor_data_store))
    return {
        "sensors": sensor_ids,
        "count": len(sensor_ids)
    }

async def send_to_kinesis(reading: SensorReading):
    """Send sensor data to Kinesis stream"""
    try:
        # This would integrate with your Kinesis ingester
        logger.info(f"Would send to Kinesis: {reading.sensor_id}")
    except Exception as e:
        logger.error(f"Error sending to Kinesis: {e}")

def train_model_background():
    """Background task to train the ML model"""
    try:
        logger.info("Starting background model training...")
        
        # Convert data to DataFrame
        df = pd.DataFrame(sensor_data_store)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Train model
        metrics = ml_model.train(df)
        
        # Save model
        ml_model.save_model("temperature_model.joblib")
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)