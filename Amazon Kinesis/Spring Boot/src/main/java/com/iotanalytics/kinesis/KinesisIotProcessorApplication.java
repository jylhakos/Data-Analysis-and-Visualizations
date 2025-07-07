package com.iotanalytics.kinesis;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Main Spring Boot Application for IoT Data Processing with Amazon Kinesis
 * 
 * This application provides:
 * - MQTT data ingestion from IoT temperature sensors
 * - Amazon Kinesis stream processing with KCL
 * - Machine Learning integration for temperature forecasting
 * - RESTful API for forecast retrieval
 * - DynamoDB for data persistence
 * - SQS for message queuing
 */
@SpringBootApplication
@EnableAsync
@EnableScheduling
public class KinesisIotProcessorApplication {

    public static void main(String[] args) {
        SpringApplication.run(KinesisIotProcessorApplication.class, args);
    }
}
