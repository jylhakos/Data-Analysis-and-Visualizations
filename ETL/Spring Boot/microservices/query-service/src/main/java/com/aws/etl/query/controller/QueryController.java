package com.aws.etl.query.controller;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.aws.etl.models.WeatherMeasurement;
import com.aws.etl.query.service.WeatherQueryService;

/**
 * REST controller for weather data queries.
 */
@RestController
@RequestMapping("/api/query")
@CrossOrigin(origins = "*") // For frontend integration
public class QueryController {

    @Autowired
    private WeatherQueryService queryService;

    @GetMapping("/latest")
    public ResponseEntity<Map<String, WeatherMeasurement>> getLatestMeasurements() {
        return ResponseEntity.ok(queryService.getLatestMeasurements());
    }

    @GetMapping("/station/{stationId}")
    public ResponseEntity<List<WeatherMeasurement>> getMeasurementsByStation(
            @PathVariable String stationId,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant from,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) Instant to) {
        
        return ResponseEntity.ok(queryService.getMeasurementsByStation(stationId, from, to));
    }

    @GetMapping("/average-temperature")
    public ResponseEntity<Map<String, Double>> getAverageTemperature() {
        return ResponseEntity.ok(queryService.getAverageTemperatureLast24Hours());
    }

    @GetMapping("/trend/{stationId}")
    public ResponseEntity<List<Map<String, Object>>> getTemperatureTrend(
            @PathVariable String stationId,
            @RequestParam(defaultValue = "24") int hours) {
        
        return ResponseEntity.ok(queryService.getTemperatureTrend(stationId, hours));
    }

    @GetMapping("/summary")
    public ResponseEntity<Map<String, Object>> getWeatherSummary() {
        return ResponseEntity.ok(queryService.getWeatherSummary());
    }

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        Map<String, Object> status = Map.of(
            "service", "query-service",
            "status", "running",
            "totalMeasurements", queryService.getTotalMeasurements()
        );
        
        return ResponseEntity.ok(status);
    }
}
