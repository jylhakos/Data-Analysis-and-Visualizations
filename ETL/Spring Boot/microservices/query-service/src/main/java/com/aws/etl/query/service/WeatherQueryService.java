package com.aws.etl.query.service;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.aws.etl.models.WeatherMeasurement;

/**
 * Query Service for weather data using simulated Athena queries.
 * In production, this would integrate with Amazon Athena and Apache Iceberg.
 */
@Service
public class WeatherQueryService {

    private static final Logger logger = LoggerFactory.getLogger(WeatherQueryService.class);

    @Value("${athena.database:weather_db}")
    private String database;

    @Value("${athena.table:weather_measurements}")
    private String table;

    // Simulated data store (in production, this would be Athena + S3)
    private final List<WeatherMeasurement> dataStore = Collections.synchronizedList(new ArrayList<>());

    /**
     * Add measurement to the simulated data store.
     */
    public void addMeasurement(WeatherMeasurement measurement) {
        dataStore.add(measurement);
        logger.debug("Added measurement from station {} to query store", measurement.getStationId());
    }

    /**
     * Get latest measurements for all stations.
     */
    public Map<String, WeatherMeasurement> getLatestMeasurements() {
        logger.info("Querying latest measurements from all stations");

        return dataStore.stream()
                .collect(Collectors.groupingBy(
                        WeatherMeasurement::getStationId,
                        Collectors.maxBy(Comparator.comparing(WeatherMeasurement::getTimestamp))
                ))
                .entrySet().stream()
                .filter(entry -> entry.getValue().isPresent())
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue().get()
                ));
    }

    /**
     * Get measurements for a specific station within a time range.
     */
    public List<WeatherMeasurement> getMeasurementsByStation(String stationId, Instant from, Instant to) {
        logger.info("Querying measurements for station {} from {} to {}", stationId, from, to);

        return dataStore.stream()
                .filter(m -> m.getStationId().equals(stationId))
                .filter(m -> m.getTimestamp().isAfter(from) && m.getTimestamp().isBefore(to))
                .sorted(Comparator.comparing(WeatherMeasurement::getTimestamp))
                .collect(Collectors.toList());
    }

    /**
     * Get average temperature by station for the last 24 hours.
     */
    public Map<String, Double> getAverageTemperatureLast24Hours() {
        Instant yesterday = Instant.now().minus(24, ChronoUnit.HOURS);
        
        logger.info("Calculating average temperature for last 24 hours");

        return dataStore.stream()
                .filter(m -> m.getTimestamp().isAfter(yesterday))
                .filter(m -> m.getTemperature() != null)
                .collect(Collectors.groupingBy(
                        WeatherMeasurement::getStationId,
                        Collectors.averagingDouble(WeatherMeasurement::getTemperature)
                ));
    }

    /**
     * Get temperature trends for a station.
     */
    public List<Map<String, Object>> getTemperatureTrend(String stationId, int hours) {
        Instant since = Instant.now().minus(hours, ChronoUnit.HOURS);
        
        logger.info("Getting temperature trend for station {} over last {} hours", stationId, hours);

        return dataStore.stream()
                .filter(m -> m.getStationId().equals(stationId))
                .filter(m -> m.getTimestamp().isAfter(since))
                .filter(m -> m.getTemperature() != null)
                .sorted(Comparator.comparing(WeatherMeasurement::getTimestamp))
                .map(m -> {
                    Map<String, Object> point = new HashMap<>();
                    point.put("timestamp", m.getTimestamp());
                    point.put("temperature", m.getTemperature());
                    point.put("humidity", m.getHumidity());
                    point.put("pressure", m.getPressure());
                    return point;
                })
                .collect(Collectors.toList());
    }

    /**
     * Get weather summary statistics.
     */
    public Map<String, Object> getWeatherSummary() {
        logger.info("Generating weather summary statistics");

        Map<String, Object> summary = new HashMap<>();
        
        if (dataStore.isEmpty()) {
            summary.put("totalMeasurements", 0);
            summary.put("stationCount", 0);
            return summary;
        }

        summary.put("totalMeasurements", dataStore.size());
        summary.put("stationCount", dataStore.stream()
                .map(WeatherMeasurement::getStationId)
                .collect(Collectors.toSet()).size());

        // Temperature statistics
        DoubleSummaryStatistics tempStats = dataStore.stream()
                .filter(m -> m.getTemperature() != null)
                .mapToDouble(WeatherMeasurement::getTemperature)
                .summaryStatistics();

        if (tempStats.getCount() > 0) {
            Map<String, Object> temperature = new HashMap<>();
            temperature.put("min", tempStats.getMin());
            temperature.put("max", tempStats.getMax());
            temperature.put("average", tempStats.getAverage());
            summary.put("temperature", temperature);
        }

        // Latest measurements timestamp
        Optional<Instant> latestTimestamp = dataStore.stream()
                .map(WeatherMeasurement::getTimestamp)
                .max(Instant::compareTo);
        
        latestTimestamp.ifPresent(timestamp -> summary.put("latestMeasurement", timestamp));

        return summary;
    }

    public int getTotalMeasurements() {
        return dataStore.size();
    }
}
