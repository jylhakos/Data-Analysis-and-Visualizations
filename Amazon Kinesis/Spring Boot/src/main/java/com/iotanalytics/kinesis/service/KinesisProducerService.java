package com.iotanalytics.kinesis.service;

import com.iotanalytics.kinesis.model.TemperatureData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.stream.function.StreamBridge;
import org.springframework.stereotype.Service;

/**
 * Service for publishing temperature data to Kinesis streams
 */
@Service
public class KinesisProducerService {

    private static final Logger logger = LoggerFactory.getLogger(KinesisProducerService.class);
    
    @Autowired
    private StreamBridge streamBridge;

    /**
     * Publishes temperature data to the Kinesis stream
     */
    public void publishTemperatureData(TemperatureData temperatureData) {
        try {
            logger.debug("Publishing temperature data: {}", temperatureData);
            
            // Send to Kinesis via Spring Cloud Stream
            boolean sent = streamBridge.send("temperatureData-out-0", temperatureData);
            
            if (sent) {
                logger.info("Successfully published temperature data for sensor: {}", temperatureData.getSensorId());
            } else {
                logger.error("Failed to publish temperature data for sensor: {}", temperatureData.getSensorId());
            }
            
        } catch (Exception e) {
            logger.error("Error publishing temperature data for sensor: {}", temperatureData.getSensorId(), e);
            throw new RuntimeException("Failed to publish temperature data", e);
        }
    }

    /**
     * Publishes batch of temperature data
     */
    public void publishTemperatureDataBatch(Iterable<TemperatureData> temperatureDataList) {
        try {
            temperatureDataList.forEach(this::publishTemperatureData);
            logger.info("Successfully published batch of temperature data");
        } catch (Exception e) {
            logger.error("Error publishing batch of temperature data", e);
            throw new RuntimeException("Failed to publish batch temperature data", e);
        }
    }
}
