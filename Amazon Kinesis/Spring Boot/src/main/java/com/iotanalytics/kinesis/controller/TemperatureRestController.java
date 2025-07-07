package com.iotanalytics.kinesis.controller;

import com.iotanalytics.kinesis.model.TemperatureData;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for temperature data and forecasting APIs
 */
@RestController
@RequestMapping("/api/temperature")
@CrossOrigin(origins = "*")
public class TemperatureRestController {

    private final Map<String, List<TemperatureData>> sensorDataStorage = new HashMap<>();

    /**
     * Get latest temperature data for a sensor
     */
    @GetMapping("/latest/{sensorId}")
    public ResponseEntity<TemperatureData> getLatestTemperature(@PathVariable String sensorId) {
        List<TemperatureData> sensorData = sensorDataStorage.get(sensorId);
        if (sensorData == null || sensorData.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        
        TemperatureData latest = sensorData.get(sensorData.size() - 1);
        return ResponseEntity.ok(latest);
    }

    /**
     * Get historical temperature data for a sensor
     */
    @GetMapping("/history/{sensorId}")
    public ResponseEntity<List<TemperatureData>> getHistoricalTemperature(
            @PathVariable String sensorId,
            @RequestParam(required = false, defaultValue = "100") int limit) {
        
        List<TemperatureData> sensorData = sensorDataStorage.get(sensorId);
        if (sensorData == null) {
            return ResponseEntity.ok(new ArrayList<>());
        }
        
        int fromIndex = Math.max(0, sensorData.size() - limit);
        List<TemperatureData> limitedData = sensorData.subList(fromIndex, sensorData.size());
        
        return ResponseEntity.ok(limitedData);
    }

    /**
     * Get all available sensors
     */
    @GetMapping("/sensors")
    public ResponseEntity<Map<String, Object>> getAllSensors() {
        Map<String, Object> response = new HashMap<>();
        response.put("sensors", sensorDataStorage.keySet());
        response.put("count", sensorDataStorage.size());
        
        return ResponseEntity.ok(response);
    }

    /**
     * Submit temperature data (for testing)
     */
    @PostMapping("/data")
    public ResponseEntity<Map<String, Object>> submitTemperatureData(@RequestBody TemperatureData temperatureData) {
        if (temperatureData.getTimestamp() == null) {
            temperatureData.setTimestamp(LocalDateTime.now());
        }
        
        String sensorId = temperatureData.getSensorId();
        sensorDataStorage.computeIfAbsent(sensorId, k -> new ArrayList<>()).add(temperatureData);
        
        // Keep only last 1000 records per sensor
        List<TemperatureData> sensorData = sensorDataStorage.get(sensorId);
        if (sensorData.size() > 1000) {
            sensorDataStorage.put(sensorId, sensorData.subList(sensorData.size() - 1000, sensorData.size()));
        }
        
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "Temperature data received");
        response.put("sensor_id", sensorId);
        response.put("timestamp", temperatureData.getTimestamp());
        
        return ResponseEntity.ok(response);
    }

    /**
     * Get sensor statistics
     */
    @GetMapping("/stats/{sensorId}")
    public ResponseEntity<Map<String, Object>> getSensorStats(@PathVariable String sensorId) {
        List<TemperatureData> sensorData = sensorDataStorage.get(sensorId);
        if (sensorData == null || sensorData.isEmpty()) {
            return ResponseEntity.notFound().build();
        }
        
        // Calculate basic statistics
        double sum = sensorData.stream().mapToDouble(TemperatureData::getTemperature).sum();
        double avg = sum / sensorData.size();
        double min = sensorData.stream().mapToDouble(TemperatureData::getTemperature).min().orElse(0);
        double max = sensorData.stream().mapToDouble(TemperatureData::getTemperature).max().orElse(0);
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("sensor_id", sensorId);
        stats.put("data_points", sensorData.size());
        stats.put("average_temperature", Math.round(avg * 100.0) / 100.0);
        stats.put("min_temperature", min);
        stats.put("max_temperature", max);
        stats.put("latest_reading", sensorData.get(sensorData.size() - 1));
        
        return ResponseEntity.ok(stats);
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "UP");
        health.put("timestamp", LocalDateTime.now());
        health.put("active_sensors", sensorDataStorage.size());
        
        return ResponseEntity.ok(health);
    }
}
