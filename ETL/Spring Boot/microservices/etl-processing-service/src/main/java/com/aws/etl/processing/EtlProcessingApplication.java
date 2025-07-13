package com.aws.etl.processing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * ETL Processing Service Application
 * 
 * This microservice handles:
 * - AWS Glue job management
 * - Batch ETL operations
 * - Data transformation and validation
 * - Integration with data lake
 */
@SpringBootApplication(scanBasePackages = {"com.aws.etl.processing", "com.aws.etl.utils"})
@EnableScheduling
public class EtlProcessingApplication {

    public static void main(String[] args) {
        SpringApplication.run(EtlProcessingApplication.class, args);
    }
}
