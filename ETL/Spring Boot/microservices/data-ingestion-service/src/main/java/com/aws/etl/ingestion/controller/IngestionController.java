package com.aws.etl.ingestion.controller;

import com.aws.etl.ingestion.service.MqttClientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

/**
 * REST controller for data ingestion service operations and monitoring.
 */
@RestController
@RequestMapping("/api/ingestion")
public class IngestionController {

    @Autowired
    private MqttClientService mqttClientService;

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("service", "data-ingestion-service");
        status.put("status", "running");
        status.put("mqttConnected", mqttClientService.isConnected());
        
        return ResponseEntity.ok(status);
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        Map<String, String> health = new HashMap<>();
        
        if (mqttClientService.isConnected()) {
            health.put("status", "UP");
            health.put("mqtt", "CONNECTED");
        } else {
            health.put("status", "DOWN");
            health.put("mqtt", "DISCONNECTED");
        }
        
        return ResponseEntity.ok(health);
    }
}
