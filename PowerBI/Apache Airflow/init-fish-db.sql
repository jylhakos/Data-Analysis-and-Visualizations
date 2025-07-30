-- Fish Weight Prediction Database Initialization
-- Creates additional database and tables for fish prediction storage

-- Create fish predictions database
CREATE DATABASE IF NOT EXISTS fish_predictions;

-- Connect to fish predictions database
\c fish_predictions;

-- Create fish predictions table
CREATE TABLE IF NOT EXISTS fish_predictions (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    species VARCHAR(50) NOT NULL,
    length1 DECIMAL(8,3) NOT NULL,
    length2 DECIMAL(8,3) NOT NULL,
    length3 DECIMAL(8,3) NOT NULL,
    height DECIMAL(8,3) NOT NULL,
    width DECIMAL(8,3) NOT NULL,
    predicted_weight DECIMAL(10,3) NOT NULL,
    confidence_score DECIMAL(5,3) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    api_version VARCHAR(20) DEFAULT '1.0.0'
);

-- Create fish model metrics table
CREATE TABLE IF NOT EXISTS fish_model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    r2_score DECIMAL(8,6),
    rmse DECIMAL(10,3),
    mae DECIMAL(10,3),
    mape DECIMAL(8,3),
    species_count INTEGER,
    training_samples INTEGER,
    model_file_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create fish dataset metadata table  
CREATE TABLE IF NOT EXISTS fish_dataset_info (
    id SERIAL PRIMARY KEY,
    dataset_version VARCHAR(50),
    total_fish_count INTEGER,
    species_distribution JSONB,
    weight_range_min DECIMAL(10,3),
    weight_range_max DECIMAL(10,3),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fish predictions
CREATE INDEX IF NOT EXISTS idx_fish_predictions_species ON fish_predictions(species);
CREATE INDEX IF NOT EXISTS idx_fish_predictions_created_at ON fish_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_fish_predictions_prediction_id ON fish_predictions(prediction_id);

-- Create indexes for fish model metrics
CREATE INDEX IF NOT EXISTS idx_fish_model_metrics_model_name ON fish_model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_fish_model_metrics_training_date ON fish_model_metrics(training_date);

-- Insert initial fish dataset metadata
INSERT INTO fish_dataset_info (
    dataset_version, 
    total_fish_count, 
    species_distribution,
    weight_range_min,
    weight_range_max
) VALUES (
    '1.0.0',
    159,
    '{"Bream": 35, "Roach": 20, "Whitefish": 17, "Parkki": 11, "Perch": 56, "Pike": 17, "Smelt": 14}',
    0.0,
    1650.0
) ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;
