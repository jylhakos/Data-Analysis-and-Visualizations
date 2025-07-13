package com.aws.etl.processing.controller;

import com.aws.etl.processing.service.EtlJobService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * REST controller for ETL processing operations.
 */
@RestController
@RequestMapping("/api/etl")
public class EtlController {

    @Autowired
    private EtlJobService etlJobService;

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("service", "etl-processing-service");
        status.put("status", "running");
        status.put("pendingMeasurements", etlJobService.getPendingCount());
        
        return ResponseEntity.ok(status);
    }

    @PostMapping("/trigger")
    public ResponseEntity<Map<String, String>> triggerBatch() {
        etlJobService.processBatch();
        
        Map<String, String> response = new HashMap<>();
        response.put("message", "ETL batch processing triggered");
        
        return ResponseEntity.ok(response);
    }
}
