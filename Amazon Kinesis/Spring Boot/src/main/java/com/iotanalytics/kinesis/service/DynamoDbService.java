package com.iotanalytics.kinesis.service;

import com.iotanalytics.kinesis.model.TemperatureData;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for DynamoDB operations
 */
public class DynamoDbService {

    private final Map<String, List<TemperatureData>> temperatureDataStore;

    public DynamoDbService() {
        this.temperatureDataStore = new HashMap<>();
    }

    /**
     * Stores temperature data in DynamoDB
     */
    public void storeTemperatureData(TemperatureData temperatureData) {
        try {
            String sensorId = temperatureData.getSensorId();
            temperatureDataStore.computeIfAbsent(sensorId, k -> new ArrayList<>()).add(temperatureData);
            System.out.println("Stored temperature data for sensor: " + sensorId);
        } catch (Exception e) {
            System.err.println("Error storing temperature data: " + e.getMessage());
            throw new RuntimeException("Failed to store temperature data", e);
        }
    }

    /**
     * Retrieves temperature data by time range
     */
    public List<TemperatureData> getTemperatureDataByTimeRange(String sensorId, LocalDateTime startDate, LocalDateTime endDate) {
        List<TemperatureData> allData = temperatureDataStore.getOrDefault(sensorId, new ArrayList<>());
        List<TemperatureData> filteredData = new ArrayList<>();
        
        for (TemperatureData data : allData) {
            if (data.getTimestamp().isAfter(startDate) && data.getTimestamp().isBefore(endDate)) {
                filteredData.add(data);
            }
        }
        
        return filteredData;
    }

    /**
     * Retrieves latest temperature data for a sensor
     */
    public TemperatureData getLatestTemperatureData(String sensorId) {
        List<TemperatureData> sensorData = temperatureDataStore.get(sensorId);
        if (sensorData == null || sensorData.isEmpty()) {
            return null;
        }
        return sensorData.get(sensorData.size() - 1);
    }

    /**
     * Retrieves all sensors
     */
    public List<String> getAllSensorIds() {
        return new ArrayList<>(temperatureDataStore.keySet());
    }

    /**
     * Gets count of data points for a sensor
     */
    public long getDataPointCount(String sensorId) {
        List<TemperatureData> sensorData = temperatureDataStore.get(sensorId);
        return sensorData != null ? sensorData.size() : 0;
    }
}
