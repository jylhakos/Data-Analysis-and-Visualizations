package com.iotanalytics.kinesis.model;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Represents temperature forecast data from ML service
 */
public class TemperatureForecast {
    
    private String sensorId;
    private String location;
    private LocalDateTime forecastDate;
    private List<ForecastPoint> dailyForecasts;
    private String modelVersion;
    private Double confidence;
    private LocalDateTime createdAt;

    public TemperatureForecast() {
        this.createdAt = LocalDateTime.now();
    }

    public TemperatureForecast(String sensorId, String location, LocalDateTime forecastDate,
                              List<ForecastPoint> dailyForecasts, String modelVersion, Double confidence) {
        this.sensorId = sensorId;
        this.location = location;
        this.forecastDate = forecastDate;
        this.dailyForecasts = dailyForecasts;
        this.modelVersion = modelVersion;
        this.confidence = confidence;
        this.createdAt = LocalDateTime.now();
    }

    // Getters and Setters
    public String getSensorId() {
        return sensorId;
    }

    public void setSensorId(String sensorId) {
        this.sensorId = sensorId;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public LocalDateTime getForecastDate() {
        return forecastDate;
    }

    public void setForecastDate(LocalDateTime forecastDate) {
        this.forecastDate = forecastDate;
    }

    public List<ForecastPoint> getDailyForecasts() {
        return dailyForecasts;
    }

    public void setDailyForecasts(List<ForecastPoint> dailyForecasts) {
        this.dailyForecasts = dailyForecasts;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    /**
     * Represents a single forecast point
     */
    public static class ForecastPoint {
        private LocalDateTime timestamp;
        private Double temperature;
        private Double minTemperature;
        private Double maxTemperature;
        private Double confidence;

        public ForecastPoint() {}

        public ForecastPoint(LocalDateTime timestamp, Double temperature, 
                           Double minTemperature, Double maxTemperature, Double confidence) {
            this.timestamp = timestamp;
            this.temperature = temperature;
            this.minTemperature = minTemperature;
            this.maxTemperature = maxTemperature;
            this.confidence = confidence;
        }

        // Getters and Setters
        public LocalDateTime getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(LocalDateTime timestamp) {
            this.timestamp = timestamp;
        }

        public Double getTemperature() {
            return temperature;
        }

        public void setTemperature(Double temperature) {
            this.temperature = temperature;
        }

        public Double getMinTemperature() {
            return minTemperature;
        }

        public void setMinTemperature(Double minTemperature) {
            this.minTemperature = minTemperature;
        }

        public Double getMaxTemperature() {
            return maxTemperature;
        }

        public void setMaxTemperature(Double maxTemperature) {
            this.maxTemperature = maxTemperature;
        }

        public Double getConfidence() {
            return confidence;
        }

        public void setConfidence(Double confidence) {
            this.confidence = confidence;
        }
    }
}
