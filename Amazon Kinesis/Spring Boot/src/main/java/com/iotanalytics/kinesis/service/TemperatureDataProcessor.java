package com.iotanalytics.kinesis.service;

import com.iotanalytics.kinesis.model.TemperatureData;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service for processing and storing temperature data
 */
public class TemperatureDataProcessor {

    private final DynamoDbService dynamoDbService;
    private final SqsService sqsService;

    public TemperatureDataProcessor(DynamoDbService dynamoDbService, SqsService sqsService) {
        this.dynamoDbService = dynamoDbService;
        this.sqsService = sqsService;
    }

    /**
     * Processes temperature data from IoT sensors
     */
    public void processTemperatureData(TemperatureData temperatureData) {
        try {
            // Validate data
            validateTemperatureData(temperatureData);
            
            // Store in DynamoDB
            dynamoDbService.storeTemperatureData(temperatureData);
            
            // Send to SQS for further processing
            sqsService.sendMessage("temperature-processing", temperatureData);
            
            System.out.println("Successfully processed temperature data for sensor: " + temperatureData.getSensorId());
            
        } catch (Exception e) {
            System.err.println("Error processing temperature data: " + e.getMessage());
            throw new RuntimeException("Failed to process temperature data", e);
        }
    }

    /**
     * Processes batch of temperature data
     */
    public void processTemperatureDataBatch(List<TemperatureData> temperatureDataList) {
        temperatureDataList.forEach(this::processTemperatureData);
    }

    /**
     * Validates temperature data
     */
    private void validateTemperatureData(TemperatureData temperatureData) {
        if (temperatureData == null) {
            throw new IllegalArgumentException("Temperature data cannot be null");
        }
        
        if (temperatureData.getSensorId() == null || temperatureData.getSensorId().trim().isEmpty()) {
            throw new IllegalArgumentException("Sensor ID cannot be null or empty");
        }
        
        if (temperatureData.getTemperature() == null) {
            throw new IllegalArgumentException("Temperature value cannot be null");
        }
        
        if (temperatureData.getTimestamp() == null) {
            temperatureData.setTimestamp(LocalDateTime.now());
        }
        
        // Validate temperature range (reasonable limits)
        double temp = temperatureData.getTemperature();
        if (temp < -100 || temp > 100) {
            throw new IllegalArgumentException("Temperature value out of reasonable range: " + temp);
        }
    }

    /**
     * Retrieves historical temperature data for ML training
     */
    public List<TemperatureData> getHistoricalData(String sensorId, LocalDateTime startDate, LocalDateTime endDate) {
        return dynamoDbService.getTemperatureDataByTimeRange(sensorId, startDate, endDate);
    }
}
