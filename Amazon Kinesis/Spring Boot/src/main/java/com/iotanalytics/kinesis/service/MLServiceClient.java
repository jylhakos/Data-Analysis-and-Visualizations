package com.iotanalytics.kinesis.service;

import com.iotanalytics.kinesis.model.TemperatureData;
import com.iotanalytics.kinesis.model.TemperatureForecast;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Client service for communicating with ML service for temperature forecasting
 */
@Service
public class MLServiceClient {

    private final RestTemplate restTemplate;
    private final String mlServiceBaseUrl;

    public MLServiceClient() {
        this.restTemplate = new RestTemplate();
        this.mlServiceBaseUrl = "http://localhost:8001"; // Will be configurable
    }

    /**
     * Sends temperature data to ML service for analysis
     */
    public void sendTemperatureDataForAnalysis(TemperatureData temperatureData) {
        try {
            String url = mlServiceBaseUrl + "/api/temperature/analyze";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<TemperatureData> request = new HttpEntity<>(temperatureData, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, request, String.class);
            
            System.out.println("ML service response: " + response.getBody());
            
        } catch (Exception e) {
            System.err.println("Error sending data to ML service: " + e.getMessage());
        }
    }

    /**
     * Requests temperature forecast from ML service
     */
    public TemperatureForecast requestTemperatureForecast(String sensorId, int daysAhead) {
        try {
            String url = mlServiceBaseUrl + "/api/forecast/temperature";
            
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("sensor_id", sensorId);
            requestBody.put("days_ahead", daysAhead);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<TemperatureForecast> response = restTemplate.exchange(
                url, HttpMethod.POST, request, TemperatureForecast.class);
            
            return response.getBody();
            
        } catch (Exception e) {
            System.err.println("Error requesting forecast from ML service: " + e.getMessage());
            return null;
        }
    }

    /**
     * Sends historical data for ML model training
     */
    public void sendHistoricalDataForTraining(List<TemperatureData> historicalData) {
        try {
            String url = mlServiceBaseUrl + "/api/model/train";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<List<TemperatureData>> request = new HttpEntity<>(historicalData, headers);
            
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, request, String.class);
            
            System.out.println("ML training response: " + response.getBody());
            
        } catch (Exception e) {
            System.err.println("Error sending training data to ML service: " + e.getMessage());
        }
    }

    /**
     * Checks ML service health
     */
    public boolean isMLServiceHealthy() {
        try {
            String url = mlServiceBaseUrl + "/health";
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            return response.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }
}
