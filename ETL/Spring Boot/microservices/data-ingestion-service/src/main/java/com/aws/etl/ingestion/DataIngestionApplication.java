package com.aws.etl.ingestion;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Data Ingestion Service Application
 * 
 * This microservice handles:
 * - MQTT message consumption from weather stations
 * - Data validation and transformation
 * - Publishing to Kafka for downstream processing
 * - Real-time monitoring and health checks
 */
@SpringBootApplication(scanBasePackages = {"com.aws.etl.ingestion", "com.aws.etl.utils"})
@EnableScheduling
public class DataIngestionApplication {

    public static void main(String[] args) {
        SpringApplication.run(DataIngestionApplication.class, args);
    }
}
