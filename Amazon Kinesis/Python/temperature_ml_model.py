import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemperatureForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and lag features for temperature forecasting"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:  # Hours
            df[f'temp_lag_{lag}h'] = df['temperature'].shift(lag)
        
        # Rolling window features
        for window in [3, 6, 12, 24]:  # Hours
            df[f'temp_rolling_mean_{window}h'] = df['temperature'].rolling(window=window).mean()
            df[f'temp_rolling_std_{window}h'] = df['temperature'].rolling(window=window).std()
        
        # Difference features
        df['temp_diff_1h'] = df['temperature'].diff(1)
        df['temp_diff_24h'] = df['temperature'].diff(24)
        
        # Weather pattern features
        if 'humidity' in df.columns:
            df['humidity_lag_1h'] = df['humidity'].shift(1)
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/prediction"""
        # Create features
        df_features = self.create_features(df)
        
        # Select feature columns (exclude target and non-feature columns)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['temperature', 'timestamp', 'sensor_id', 'location']]
        
        # Remove rows with NaN values (due to lag features)
        df_clean = df_features.dropna()
        
        if df_clean.empty:
            raise ValueError("No valid data after feature engineering")
        
        X = df_clean[feature_cols].values
        y = df_clean['temperature'].values
        
        return X, y, feature_cols
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the temperature forecasting model"""
        logger.info("Starting model training...")
        
        # Prepare data
        X, y, self.feature_cols = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        self.is_trained = True
        logger.info(f"Model training completed. Test MAE: {metrics['test_mae']:.2f}°C")
        
        return metrics
    
    def predict_next_24h(self, recent_data: pd.DataFrame) -> pd.DataFrame:
        """Predict temperature for the next 24 hours"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_time = recent_data['timestamp'].max()
        
        # Prepare recent data with features
        df_with_features = self.create_features(recent_data)
        
        for hour in range(1, 25):  # Predict next 24 hours
            pred_time = current_time + timedelta(hours=hour)
            
            # Create feature row for prediction time
            pred_row = self._create_prediction_features(df_with_features, pred_time)
            
            if pred_row is not None:
                # Scale features
                pred_scaled = self.scaler.transform(pred_row.reshape(1, -1))
                
                # Make prediction
                pred_temp = self.model.predict(pred_scaled)[0]
                
                predictions.append({
                    'timestamp': pred_time,
                    'predicted_temperature': round(pred_temp, 2),
                    'prediction_hour': hour
                })
        
        return pd.DataFrame(predictions)
    
    def _create_prediction_features(self, df: pd.DataFrame, pred_time: datetime) -> Optional[np.ndarray]:
        """Create features for a specific prediction time"""
        try:
            # Create a row with time-based features
            pred_features = {}
            
            # Time-based features
            pred_features['hour'] = pred_time.hour
            pred_features['day_of_week'] = pred_time.weekday()
            pred_features['month'] = pred_time.month
            pred_features['day_of_year'] = pred_time.timetuple().tm_yday
            
            # Cyclical encoding
            pred_features['hour_sin'] = np.sin(2 * np.pi * pred_time.hour / 24)
            pred_features['hour_cos'] = np.cos(2 * np.pi * pred_time.hour / 24)
            pred_features['day_sin'] = np.sin(2 * np.pi * pred_time.weekday() / 7)
            pred_features['day_cos'] = np.cos(2 * np.pi * pred_time.weekday() / 7)
            pred_features['month_sin'] = np.sin(2 * np.pi * pred_time.month / 12)
            pred_features['month_cos'] = np.cos(2 * np.pi * pred_time.month / 12)
            
            # Get latest values for lag and rolling features
            latest_temps = df['temperature'].tail(24).values  # Last 24 hours
            latest_humidity = df['humidity'].tail(24).values if 'humidity' in df.columns else None
            
            # Lag features (use available data)
            for lag in [1, 2, 3, 6, 12, 24]:
                if len(latest_temps) >= lag:
                    pred_features[f'temp_lag_{lag}h'] = latest_temps[-lag]
                else:
                    pred_features[f'temp_lag_{lag}h'] = latest_temps[-1]  # Use last available
            
            # Rolling features
            for window in [3, 6, 12, 24]:
                if len(latest_temps) >= window:
                    pred_features[f'temp_rolling_mean_{window}h'] = np.mean(latest_temps[-window:])
                    pred_features[f'temp_rolling_std_{window}h'] = np.std(latest_temps[-window:])
                else:
                    pred_features[f'temp_rolling_mean_{window}h'] = np.mean(latest_temps)
                    pred_features[f'temp_rolling_std_{window}h'] = np.std(latest_temps)
            
            # Difference features
            if len(latest_temps) >= 2:
                pred_features['temp_diff_1h'] = latest_temps[-1] - latest_temps[-2]
            else:
                pred_features['temp_diff_1h'] = 0
                
            if len(latest_temps) >= 24:
                pred_features['temp_diff_24h'] = latest_temps[-1] - latest_temps[-24]
            else:
                pred_features['temp_diff_24h'] = 0
            
            # Humidity features
            if latest_humidity is not None and len(latest_humidity) > 0:
                pred_features['humidity_lag_1h'] = latest_humidity[-1]
                pred_features['temp_humidity_ratio'] = latest_temps[-1] / (latest_humidity[-1] + 1e-6)
            else:
                pred_features['humidity_lag_1h'] = 50.0  # Default humidity
                pred_features['temp_humidity_ratio'] = latest_temps[-1] / 50.0
            
            # Create feature array in the same order as training
            feature_array = np.array([pred_features.get(col, 0) for col in self.feature_cols])
            return feature_array
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365)) + 
                      5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + 
                      np.random.normal(0, 2, len(dates)),
        'humidity': 50 + 20 * np.random.random(len(dates)),
        'sensor_id': 'sensor_001'
    })
    
    # Train model
    model = TemperatureForecastModel()
    metrics = model.train(sample_data)
    print("Training metrics:", metrics)
    
    # Make predictions
    recent_data = sample_data.tail(168)  # Last week of data
    predictions = model.predict_next_24h(recent_data)
    print("\nNext 24h predictions:")
    print(predictions.head())