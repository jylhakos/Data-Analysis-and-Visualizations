#!/usr/bin/env python3
"""
Machine Learning Service for Temperature Forecasting
Uses FastAPI and scikit-learn for time series forecasting
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import uvicorn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Temperature Forecasting ML Service", version="1.0.0")

# Global variables for model and scaler
model = None
scaler = None
is_model_trained = False

# Data models
class TemperatureData(BaseModel):
    sensor_id: str
    temperature: float
    humidity: Optional[float] = None
    location: Optional[str] = None
    timestamp: str
    device_type: Optional[str] = None
    battery_level: Optional[int] = None
    signal_strength: Optional[int] = None

class ForecastRequest(BaseModel):
    sensor_id: str
    days_ahead: int = 7

class ForecastPoint(BaseModel):
    timestamp: str
    temperature: float
    min_temperature: float
    max_temperature: float
    confidence: float

class TemperatureForecast(BaseModel):
    sensor_id: str
    location: Optional[str] = None
    forecast_date: str
    daily_forecasts: List[ForecastPoint]
    model_version: str = "1.0"
    confidence: float
    created_at: str

class TrainingData(BaseModel):
    historical_data: List[TemperatureData]

# In-memory storage for temperature data (replace with real database in production)
temperature_storage = {}

def prepare_features(data_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare features for machine learning model
    """
    # Extract time-based features
    data_df['hour'] = data_df['timestamp'].dt.hour
    data_df['day_of_week'] = data_df['timestamp'].dt.dayofweek
    data_df['day_of_year'] = data_df['timestamp'].dt.dayofyear
    data_df['month'] = data_df['timestamp'].dt.month
    
    # Create lag features
    data_df = data_df.sort_values('timestamp')
    data_df['temp_lag_1'] = data_df['temperature'].shift(1)
    data_df['temp_lag_2'] = data_df['temperature'].shift(2)
    data_df['temp_lag_24'] = data_df['temperature'].shift(24)  # 24 hours ago
    
    # Rolling averages
    data_df['temp_avg_3'] = data_df['temperature'].rolling(window=3).mean()
    data_df['temp_avg_6'] = data_df['temperature'].rolling(window=6).mean()
    data_df['temp_avg_12'] = data_df['temperature'].rolling(window=12).mean()
    
    # Select features for training
    feature_columns = [
        'hour', 'day_of_week', 'day_of_year', 'month',
        'temp_lag_1', 'temp_lag_2', 'temp_lag_24',
        'temp_avg_3', 'temp_avg_6', 'temp_avg_12'
    ]
    
    if 'humidity' in data_df.columns:
        data_df['humidity'].fillna(data_df['humidity'].mean(), inplace=True)
        feature_columns.append('humidity')
    
    # Fill NaN values
    for col in feature_columns:
        if col in data_df.columns:
            data_df[col].fillna(data_df[col].mean(), inplace=True)
    
    return data_df[feature_columns].values

def train_model(sensor_id: str, temperature_data: List[TemperatureData]) -> dict:
    """
    Train the machine learning model for temperature forecasting
    """
    global model, scaler, is_model_trained
    
    try:
        # Convert to DataFrame
        data_list = []
        for temp_data in temperature_data:
            data_list.append({
                'timestamp': datetime.fromisoformat(temp_data.timestamp.replace('Z', '+00:00')),
                'temperature': temp_data.temperature,
                'humidity': temp_data.humidity,
                'sensor_id': temp_data.sensor_id
            })
        
        df = pd.DataFrame(data_list)
        
        if len(df) < 50:  # Need at least 50 data points
            raise ValueError("Insufficient data for training. Need at least 50 data points.")
        
        # Prepare features
        X = prepare_features(df.copy())
        y = df['temperature'].values
        
        # Remove rows with NaN values
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 30:
            raise ValueError("Insufficient valid data after preprocessing")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        is_model_trained = True
        
        # Save model and scaler
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/temp_forecast_model.pkl")
        joblib.dump(scaler, f"{model_dir}/temp_forecast_scaler.pkl")
        
        logger.info(f"Model trained successfully. MAE: {mae:.2f}, MSE: {mse:.2f}")
        
        return {
            "status": "success",
            "mae": mae,
            "mse": mse,
            "data_points": len(X),
            "message": f"Model trained with {len(X)} data points"
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

def load_model():
    """
    Load pre-trained model and scaler
    """
    global model, scaler, is_model_trained
    
    try:
        model_path = "models/temp_forecast_model.pkl"
        scaler_path = "models/temp_forecast_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            is_model_trained = True
            logger.info("Pre-trained model loaded successfully")
        else:
            logger.info("No pre-trained model found")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def generate_forecast(sensor_id: str, days_ahead: int) -> TemperatureForecast:
    """
    Generate temperature forecast for specified days ahead
    """
    global model, scaler, is_model_trained
    
    if not is_model_trained or model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Get historical data for the sensor
        if sensor_id not in temperature_storage:
            raise HTTPException(status_code=404, detail=f"No data found for sensor {sensor_id}")
        
        historical_data = temperature_storage[sensor_id]
        if len(historical_data) < 24:  # Need at least 24 hours of data
            raise HTTPException(status_code=400, detail="Insufficient historical data for forecasting")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': datetime.fromisoformat(d.timestamp.replace('Z', '+00:00')),
            'temperature': d.temperature,
            'humidity': d.humidity,
            'sensor_id': d.sensor_id
        } for d in historical_data])
        
        df = df.sort_values('timestamp')
        
        # Generate forecasts
        forecast_points = []
        current_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        
        for day in range(days_ahead):
            for hour in range(24):  # 24 hourly predictions per day
                forecast_time = current_time + timedelta(days=day, hours=hour)
                
                # Create feature vector for prediction
                temp_df = df.copy()
                temp_df.loc[len(temp_df)] = {
                    'timestamp': forecast_time,
                    'temperature': np.nan,
                    'humidity': df['humidity'].iloc[-1] if df['humidity'].iloc[-1] else 50.0,
                    'sensor_id': sensor_id
                }
                
                # Prepare features
                features = prepare_features(temp_df.copy())
                if len(features) > 0:
                    feature_vector = features[-1].reshape(1, -1)
                    
                    # Handle NaN values
                    if np.isnan(feature_vector).any():
                        # Use mean values for missing features
                        feature_vector = np.nan_to_num(feature_vector, 
                                                     nan=np.nanmean(features[:-1], axis=0))
                    
                    # Scale and predict
                    feature_vector_scaled = scaler.transform(feature_vector)
                    predicted_temp = model.predict(feature_vector_scaled)[0]
                    
                    # Calculate confidence (simplified)
                    confidence = max(0.6, 1.0 - (day * 0.1))  # Decreasing confidence over time
                    
                    # Add some uncertainty bounds
                    uncertainty = 1.0 + (day * 0.5)  # Increasing uncertainty over time
                    min_temp = predicted_temp - uncertainty
                    max_temp = predicted_temp + uncertainty
                    
                    forecast_points.append(ForecastPoint(
                        timestamp=forecast_time.isoformat(),
                        temperature=round(predicted_temp, 2),
                        min_temperature=round(min_temp, 2),
                        max_temperature=round(max_temp, 2),
                        confidence=round(confidence, 2)
                    ))
                    
                    # Update DataFrame with prediction for next iteration
                    df.loc[len(df)] = {
                        'timestamp': forecast_time,
                        'temperature': predicted_temp,
                        'humidity': df['humidity'].iloc[-1] if df['humidity'].iloc[-1] else 50.0,
                        'sensor_id': sensor_id
                    }
        
        # Calculate overall confidence
        overall_confidence = sum(fp.confidence for fp in forecast_points) / len(forecast_points)
        
        return TemperatureForecast(
            sensor_id=sensor_id,
            location=historical_data[0].location if historical_data[0].location else "Unknown",
            forecast_date=datetime.now().isoformat(),
            daily_forecasts=forecast_points,
            model_version="1.0",
            confidence=round(overall_confidence, 2),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Temperature Forecasting ML Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_trained": is_model_trained,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/temperature/analyze")
async def analyze_temperature(temperature_data: TemperatureData):
    """
    Store and analyze incoming temperature data
    """
    try:
        sensor_id = temperature_data.sensor_id
        
        # Store data in memory (replace with real database in production)
        if sensor_id not in temperature_storage:
            temperature_storage[sensor_id] = []
        
        temperature_storage[sensor_id].append(temperature_data)
        
        # Keep only last 1000 data points per sensor
        if len(temperature_storage[sensor_id]) > 1000:
            temperature_storage[sensor_id] = temperature_storage[sensor_id][-1000:]
        
        logger.info(f"Analyzed temperature data for sensor {sensor_id}: {temperature_data.temperature}°C")
        
        return {
            "status": "success",
            "message": f"Temperature data analyzed for sensor {sensor_id}",
            "data_points": len(temperature_storage[sensor_id])
        }
        
    except Exception as e:
        logger.error(f"Error analyzing temperature data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast/temperature")
async def forecast_temperature(request: ForecastRequest):
    """
    Generate temperature forecast for a sensor
    """
    try:
        forecast = generate_forecast(request.sensor_id, request.days_ahead)
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/train")
async def train_temperature_model(training_data: List[TemperatureData]):
    """
    Train the ML model with historical temperature data
    """
    try:
        if not training_data:
            raise HTTPException(status_code=400, detail="Training data cannot be empty")
        
        # Group by sensor and train model
        sensor_data = {}
        for data in training_data:
            if data.sensor_id not in sensor_data:
                sensor_data[data.sensor_id] = []
            sensor_data[data.sensor_id].append(data)
        
        # Train with data from the sensor with most data points
        best_sensor = max(sensor_data.keys(), key=lambda k: len(sensor_data[k]))
        result = train_model(best_sensor, sensor_data[best_sensor])
        
        # Store training data
        for sensor_id, data_list in sensor_data.items():
            temperature_storage[sensor_id] = data_list
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors")
async def get_sensors():
    """
    Get list of available sensors
    """
    return {
        "sensors": list(temperature_storage.keys()),
        "count": len(temperature_storage)
    }

@app.get("/api/sensors/{sensor_id}/data")
async def get_sensor_data(sensor_id: str, limit: int = 100):
    """
    Get recent data for a specific sensor
    """
    if sensor_id not in temperature_storage:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    
    data = temperature_storage[sensor_id][-limit:]
    return {
        "sensor_id": sensor_id,
        "data_points": len(data),
        "data": data
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Temperature Forecasting ML Service")
    load_model()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
