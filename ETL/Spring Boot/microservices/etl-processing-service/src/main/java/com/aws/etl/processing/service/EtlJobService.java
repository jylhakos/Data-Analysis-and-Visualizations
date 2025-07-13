package com.aws.etl.processing.service;

import com.aws.etl.models.WeatherMeasurement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * ETL Processing Service for batch processing of weather data.
 * In a production environment, this would integrate with AWS Glue.
 */
@Service
public class EtlJobService {

    private static final Logger logger = LoggerFactory.getLogger(EtlJobService.class);

    @Value("${etl.batch.size:100}")
    private int batchSize;

    @Value("${etl.processing.interval:300000}") // 5 minutes
    private long processingInterval;

    private final ConcurrentLinkedQueue<WeatherMeasurement> pendingMeasurements = new ConcurrentLinkedQueue<>();

    /**
     * Add a measurement to the processing queue.
     */
    public void queueMeasurement(WeatherMeasurement measurement) {
        pendingMeasurements.offer(measurement);
        logger.debug("Queued measurement from station {} for batch processing", measurement.getStationId());
    }

    /**
     * Process batch of measurements every configured interval.
     */
    @Scheduled(fixedDelayString = "${etl.processing.interval:300000}")
    public void processBatch() {
        if (pendingMeasurements.isEmpty()) {
            logger.debug("No measurements to process");
            return;
        }

        List<WeatherMeasurement> batch = new ArrayList<>();
        
        // Collect batch
        for (int i = 0; i < batchSize && !pendingMeasurements.isEmpty(); i++) {
            WeatherMeasurement measurement = pendingMeasurements.poll();
            if (measurement != null) {
                batch.add(measurement);
            }
        }

        if (!batch.isEmpty()) {
            processMeasurements(batch);
        }
    }

    private void processMeasurements(List<WeatherMeasurement> measurements) {
        logger.info("Processing batch of {} measurements", measurements.size());

        try {
            // Simulate ETL operations
            transformData(measurements);
            validateData(measurements);
            loadToDataLake(measurements);
            
            logger.info("Successfully processed batch of {} measurements", measurements.size());
            
        } catch (Exception e) {
            logger.error("Error processing batch: {}", e.getMessage(), e);
            // In a real implementation, failed measurements would be sent to a dead letter queue
        }
    }

    private void transformData(List<WeatherMeasurement> measurements) {
        // Simulate data transformation
        for (WeatherMeasurement measurement : measurements) {
            // Example transformations:
            // - Convert temperature units if needed
            // - Calculate derived metrics
            // - Enrich with geographical data
            
            if (measurement.getTimestamp() == null) {
                measurement.setTimestamp(Instant.now());
            }
        }
        
        logger.debug("Transformed {} measurements", measurements.size());
    }

    private void validateData(List<WeatherMeasurement> measurements) {
        // Simulate data validation
        for (WeatherMeasurement measurement : measurements) {
            if (measurement.getTemperature() != null && 
                (measurement.getTemperature() < -100 || measurement.getTemperature() > 100)) {
                logger.warn("Temperature out of range for station {}: {}°C", 
                           measurement.getStationId(), measurement.getTemperature());
            }
        }
        
        logger.debug("Validated {} measurements", measurements.size());
    }

    private void loadToDataLake(List<WeatherMeasurement> measurements) {
        // Simulate loading to data lake (S3 + Apache Iceberg)
        logger.info("Loading {} measurements to data lake", measurements.size());
        
        // In a real implementation, this would:
        // 1. Convert to Parquet format
        // 2. Write to S3 using Apache Iceberg table format
        // 3. Update table metadata
        // 4. Trigger Athena catalog updates
        
        for (WeatherMeasurement measurement : measurements) {
            logger.debug("Loaded measurement from station {} to data lake", measurement.getStationId());
        }
    }

    public int getPendingCount() {
        return pendingMeasurements.size();
    }
}
