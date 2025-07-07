package com.iotanalytics.kinesis.controller;

import com.iotanalytics.kinesis.model.TemperatureData;
import com.iotanalytics.kinesis.model.TemperatureForecast;
import com.iotanalytics.kinesis.service.DynamoDbService;
import com.iotanalytics.kinesis.service.MLServiceClient;
import com.iotanalytics.kinesis.service.TemperatureDataProcessor;

import java.time.LocalDateTime;
import java.util.List;

/**
 * REST Controller for temperature data and forecasting APIs
 */
public class TemperatureController {

    private final DynamoDbService dynamoDbService;
    private final MLServiceClient mlServiceClient;
    private final TemperatureDataProcessor temperatureDataProcessor;

    public TemperatureController(DynamoDbService dynamoDbService,
                               MLServiceClient mlServiceClient,
                               TemperatureDataProcessor temperatureDataProcessor) {
        this.dynamoDbService = dynamoDbService;
        this.mlServiceClient = mlServiceClient;
        this.temperatureDataProcessor = temperatureDataProcessor;
    }

    /**
     * Get latest temperature data for a sensor
     */
    public TemperatureData getLatestTemperature(String sensorId) {
        return dynamoDbService.getLatestTemperatureData(sensorId);
    }

    /**
     * Get historical temperature data for a sensor
     */
    public List<TemperatureData> getHistoricalTemperature(String sensorId, 
                                                         LocalDateTime startDate, 
                                                         LocalDateTime endDate) {
        return dynamoDbService.getTemperatureDataByTimeRange(sensorId, startDate, endDate);
    }

    /**
     * Get temperature forecast for a sensor
     */
    public TemperatureForecast getTemperatureForecast(String sensorId, int daysAhead) {
        return mlServiceClient.requestTemperatureForecast(sensorId, daysAhead);
    }

    /**
     * Get all available sensors
     */
    public List<String> getAllSensors() {
        return dynamoDbService.getAllSensorIds();
    }

    /**
     * Get sensor statistics
     */
    public SensorStats getSensorStats(String sensorId) {
        long dataPointCount = dynamoDbService.getDataPointCount(sensorId);
        TemperatureData latestData = dynamoDbService.getLatestTemperatureData(sensorId);
        
        return new SensorStats(sensorId, dataPointCount, latestData);
    }

    /**
     * Trigger ML model training with historical data
     */
    public String triggerModelTraining(String sensorId) {
        LocalDateTime endDate = LocalDateTime.now();
        LocalDateTime startDate = endDate.minusDays(30); // Last 30 days
        
        List<TemperatureData> historicalData = dynamoDbService.getTemperatureDataByTimeRange(
            sensorId, startDate, endDate);
        
        if (historicalData.isEmpty()) {
            return "No historical data available for training";
        }
        
        mlServiceClient.sendHistoricalDataForTraining(historicalData);
        return "Model training initiated with " + historicalData.size() + " data points";
    }

    /**
     * Health check endpoint
     */
    public String healthCheck() {
        boolean mlServiceHealthy = mlServiceClient.isMLServiceHealthy();
        return mlServiceHealthy ? "OK" : "ML Service Unavailable";
    }

    /**
     * Sensor statistics data class
     */
    public static class SensorStats {
        private final String sensorId;
        private final long dataPointCount;
        private final TemperatureData latestData;

        public SensorStats(String sensorId, long dataPointCount, TemperatureData latestData) {
            this.sensorId = sensorId;
            this.dataPointCount = dataPointCount;
            this.latestData = latestData;
        }

        public String getSensorId() {
            return sensorId;
        }

        public long getDataPointCount() {
            return dataPointCount;
        }

        public TemperatureData getLatestData() {
            return latestData;
        }
    }
}
