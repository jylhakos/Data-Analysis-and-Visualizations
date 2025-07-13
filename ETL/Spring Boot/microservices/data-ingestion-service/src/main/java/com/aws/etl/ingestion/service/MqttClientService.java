package com.aws.etl.ingestion.service;

import com.aws.etl.models.WeatherMeasurement;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import org.eclipse.paho.client.mqttv3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validator;
import java.time.Instant;
import java.util.Set;

/**
 * MQTT client service for receiving weather station data.
 * Validates incoming messages and processes them for further handling.
 */
@Service
public class MqttClientService implements MqttCallback {

    private static final Logger logger = LoggerFactory.getLogger(MqttClientService.class);

    @Value("${mqtt.broker.url:tcp://localhost:1883}")
    private String brokerUrl;

    @Value("${mqtt.client.id:weather-ingestion-service}")
    private String clientId;

    @Value("${mqtt.topic.weather:weather/measurements}")
    private String weatherTopic;

    private MqttClient mqttClient;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Autowired
    private Validator validator;

    @Autowired
    private MeterRegistry meterRegistry;

    private Counter messagesReceivedCounter;
    private Counter messagesValidCounter;
    private Counter messagesInvalidCounter;

    @PostConstruct
    public void initialize() {
        initializeMetrics();
        connectToMqttBroker();
    }

    private void initializeMetrics() {
        messagesReceivedCounter = Counter.builder("mqtt.messages.received")
                .description("Number of MQTT messages received")
                .register(meterRegistry);
        
        messagesValidCounter = Counter.builder("mqtt.messages.valid")
                .description("Number of valid MQTT messages")
                .register(meterRegistry);
        
        messagesInvalidCounter = Counter.builder("mqtt.messages.invalid")
                .description("Number of invalid MQTT messages")
                .register(meterRegistry);
    }

    private void connectToMqttBroker() {
        try {
            mqttClient = new MqttClient(brokerUrl, clientId);
            
            MqttConnectOptions options = new MqttConnectOptions();
            options.setCleanSession(true);
            options.setConnectionTimeout(30);
            options.setKeepAliveInterval(60);
            options.setAutomaticReconnect(true);
            
            mqttClient.setCallback(this);
            mqttClient.connect(options);
            
            // Subscribe to weather measurement topic
            mqttClient.subscribe(weatherTopic, 1);
            
            logger.info("Connected to MQTT broker at {} and subscribed to topic {}", brokerUrl, weatherTopic);
            
        } catch (MqttException e) {
            logger.error("Failed to connect to MQTT broker: {}", e.getMessage(), e);
            throw new RuntimeException("MQTT connection failed", e);
        }
    }

    @Override
    public void connectionLost(Throwable cause) {
        logger.warn("MQTT connection lost: {}", cause.getMessage());
        // Automatic reconnection is enabled in connection options
    }

    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        messagesReceivedCounter.increment();
        
        try {
            String payload = new String(message.getPayload());
            logger.debug("Received MQTT message from topic {}: {}", topic, payload);
            
            // Parse JSON to WeatherMeasurement object
            WeatherMeasurement measurement = objectMapper.readValue(payload, WeatherMeasurement.class);
            
            // Set timestamp if not provided
            if (measurement.getTimestamp() == null) {
                measurement.setTimestamp(Instant.now());
            }
            
            // Validate the measurement
            Set<ConstraintViolation<WeatherMeasurement>> violations = validator.validate(measurement);
            
            if (violations.isEmpty()) {
                messagesValidCounter.increment();
                
                // Process the valid measurement
                processWeatherMeasurement(measurement);
                
                logger.debug("Successfully processed weather measurement from station {}", 
                           measurement.getStationId());
            } else {
                messagesInvalidCounter.increment();
                logger.warn("Invalid weather measurement received: {}", 
                          violations.stream()
                                   .map(ConstraintViolation::getMessage)
                                   .reduce((a, b) -> a + ", " + b)
                                   .orElse("Unknown validation error"));
            }
            
        } catch (Exception e) {
            messagesInvalidCounter.increment();
            logger.error("Error processing MQTT message: {}", e.getMessage(), e);
        }
    }

    @Override
    public void deliveryComplete(IMqttDeliveryToken token) {
        // Not used for message consumption
    }

    private void processWeatherMeasurement(WeatherMeasurement measurement) {
        // For now, just log the measurement
        // In a full implementation, this would publish to Kafka
        logger.info("Processing weather measurement: Station={}, Temperature={}°C, Humidity={}%, Timestamp={}", 
                   measurement.getStationId(), 
                   measurement.getTemperature(), 
                   measurement.getHumidity(),
                   measurement.getTimestamp());
    }

    @PreDestroy
    public void cleanup() {
        if (mqttClient != null && mqttClient.isConnected()) {
            try {
                mqttClient.disconnect();
                mqttClient.close();
                logger.info("Disconnected from MQTT broker");
            } catch (MqttException e) {
                logger.error("Error disconnecting from MQTT broker: {}", e.getMessage(), e);
            }
        }
    }

    // Health check method
    public boolean isConnected() {
        return mqttClient != null && mqttClient.isConnected();
    }
}
