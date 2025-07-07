package com.iotanalytics.kinesis.service;

import com.iotanalytics.kinesis.model.TemperatureData;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;

/**
 * Kinesis Consumer Service using Spring Cloud Stream
 * Processes temperature data from Kinesis streams using KCL
 */
@Service
public class KinesisConsumerService {

    private final TemperatureDataProcessor temperatureDataProcessor;
    private final MLServiceClient mlServiceClient;

    public KinesisConsumerService(TemperatureDataProcessor temperatureDataProcessor,
                                 MLServiceClient mlServiceClient) {
        this.temperatureDataProcessor = temperatureDataProcessor;
        this.mlServiceClient = mlServiceClient;
    }

    /**
     * Processes incoming temperature data from Kinesis stream
     * This method is automatically called by Spring Cloud Stream KCL integration
     */
    @StreamListener("temperatureData-in-0")
    public void processTemperatureData(@Payload TemperatureData temperatureData) {
        try {
            System.out.println("Processing temperature data: " + temperatureData);
            
            // Process and store the temperature data
            temperatureDataProcessor.processTemperatureData(temperatureData);
            
            // Send to ML service for analysis and forecasting
            mlServiceClient.sendTemperatureDataForAnalysis(temperatureData);
            
        } catch (Exception e) {
            System.err.println("Error processing temperature data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Alternative function-based approach for Spring Cloud Stream 3.x
     */
    public java.util.function.Consumer<TemperatureData> temperatureDataProcessor() {
        return temperatureData -> {
            try {
                System.out.println("Processing temperature data via function: " + temperatureData);
                
                // Process and store the temperature data
                temperatureDataProcessor.processTemperatureData(temperatureData);
                
                // Send to ML service for analysis and forecasting
                mlServiceClient.sendTemperatureDataForAnalysis(temperatureData);
                
            } catch (Exception e) {
                System.err.println("Error processing temperature data: " + e.getMessage());
                e.printStackTrace();
            }
        };
    }
}
