package com.aws.etl.streaming;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Stream Processing Service Application
 * 
 * This microservice handles:
 * - Apache Flink job management
 * - Real-time stream processing
 * - Complex event processing
 * - Stream analytics
 */
@SpringBootApplication(scanBasePackages = {"com.aws.etl.streaming", "com.aws.etl.utils"})
public class StreamProcessingApplication {

    public static void main(String[] args) {
        SpringApplication.run(StreamProcessingApplication.class, args);
    }
}
