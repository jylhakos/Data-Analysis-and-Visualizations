package com.iotanalytics.kinesis.mqtt;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.iotanalytics.kinesis.model.TemperatureData;
import com.iotanalytics.kinesis.service.KinesisProducerService;
import org.eclipse.paho.client.mqttv3.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.time.LocalDateTime;

/**
 * MQTT Client for receiving IoT sensor data
 */
@Component
public class MqttTemperatureListener implements MqttCallback {

    @Value("${mqtt.broker.url}")
    private String brokerUrl;

    @Value("${mqtt.broker.client-id}")
    private String clientId;

    @Value("${mqtt.topics.temperature}")
    private String temperatureTopic;

    @Autowired
    private KinesisProducerService kinesisProducerService;

    private MqttClient mqttClient;
    private ObjectMapper objectMapper;

    @PostConstruct
    public void initialize() {
        try {
            objectMapper = new ObjectMapper();
            objectMapper.registerModule(new JavaTimeModule());
            
            mqttClient = new MqttClient(brokerUrl, clientId);
            mqttClient.setCallback(this);
            
            MqttConnectOptions options = new MqttConnectOptions();
            options.setCleanSession(true);
            options.setConnectionTimeout(30);
            options.setKeepAliveInterval(60);
            options.setAutomaticReconnect(true);
            
            mqttClient.connect(options);
            
            // Subscribe to temperature topic
            mqttClient.subscribe(temperatureTopic, 1);
            
            System.out.println("MQTT client connected and subscribed to: " + temperatureTopic);
            
        } catch (MqttException e) {
            System.err.println("Error initializing MQTT client: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @PreDestroy
    public void cleanup() {
        try {
            if (mqttClient != null && mqttClient.isConnected()) {
                mqttClient.disconnect();
                mqttClient.close();
            }
        } catch (MqttException e) {
            System.err.println("Error disconnecting MQTT client: " + e.getMessage());
        }
    }

    @Override
    public void connectionLost(Throwable cause) {
        System.err.println("MQTT connection lost: " + cause.getMessage());
        // Auto-reconnect is enabled in options
    }

    @Override
    public void messageArrived(String topic, MqttMessage message) throws Exception {
        try {
            String payload = new String(message.getPayload());
            System.out.println("Received MQTT message on topic " + topic + ": " + payload);
            
            // Parse JSON message to TemperatureData
            TemperatureData temperatureData = parseTemperatureData(payload);
            
            if (temperatureData != null) {
                // Send to Kinesis stream
                kinesisProducerService.publishTemperatureData(temperatureData);
            }
            
        } catch (Exception e) {
            System.err.println("Error processing MQTT message: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Override
    public void deliveryComplete(IMqttDeliveryToken token) {
        // Not used for incoming messages
    }

    private TemperatureData parseTemperatureData(String payload) {
        try {
            // Try to parse as JSON
            TemperatureData data = objectMapper.readValue(payload, TemperatureData.class);
            
            // Set timestamp if not provided
            if (data.getTimestamp() == null) {
                data.setTimestamp(LocalDateTime.now());
            }
            
            return data;
            
        } catch (Exception e) {
            System.err.println("Error parsing temperature data: " + e.getMessage());
            
            // Try to parse as simple temperature value
            try {
                double temperature = Double.parseDouble(payload.trim());
                
                // Extract sensor ID from topic (assuming format: sensor/temperature/SENSOR_ID)
                String[] topicParts = temperatureTopic.split("/");
                String sensorId = "unknown";
                if (topicParts.length > 2) {
                    sensorId = topicParts[topicParts.length - 1].replace("+", "default");
                }
                
                return new TemperatureData(sensorId, temperature, LocalDateTime.now());
                
            } catch (NumberFormatException nfe) {
                System.err.println("Could not parse temperature value: " + payload);
                return null;
            }
        }
    }
}
