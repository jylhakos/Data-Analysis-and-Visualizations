package com.iotanalytics.kinesis.model;

import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Represents temperature sensor data from IoT devices
 */
public class TemperatureData {
    
    private String sensorId;
    
    private Double temperature;
    
    private Double humidity;
    
    private String location;
    
    private LocalDateTime timestamp;
    
    private String deviceType;
    
    private Integer batteryLevel;
    
    private Integer signalStrength;

    // Default constructor
    public TemperatureData() {
        this.timestamp = LocalDateTime.now();
    }

    // Constructor with essential fields
    public TemperatureData(String sensorId, Double temperature, LocalDateTime timestamp) {
        this.sensorId = sensorId;
        this.temperature = temperature;
        this.timestamp = timestamp;
    }

    // Full constructor
    public TemperatureData(String sensorId, Double temperature, Double humidity, 
                          String location, LocalDateTime timestamp, String deviceType,
                          Integer batteryLevel, Integer signalStrength) {
        this.sensorId = sensorId;
        this.temperature = temperature;
        this.humidity = humidity;
        this.location = location;
        this.timestamp = timestamp;
        this.deviceType = deviceType;
        this.batteryLevel = batteryLevel;
        this.signalStrength = signalStrength;
    }

    // Getters and Setters
    public String getSensorId() {
        return sensorId;
    }

    public void setSensorId(String sensorId) {
        this.sensorId = sensorId;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Double getHumidity() {
        return humidity;
    }

    public void setHumidity(Double humidity) {
        this.humidity = humidity;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }

    public String getDeviceType() {
        return deviceType;
    }

    public void setDeviceType(String deviceType) {
        this.deviceType = deviceType;
    }

    public Integer getBatteryLevel() {
        return batteryLevel;
    }

    public void setBatteryLevel(Integer batteryLevel) {
        this.batteryLevel = batteryLevel;
    }

    public Integer getSignalStrength() {
        return signalStrength;
    }

    public void setSignalStrength(Integer signalStrength) {
        this.signalStrength = signalStrength;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TemperatureData that = (TemperatureData) o;
        return Objects.equals(sensorId, that.sensorId) &&
               Objects.equals(temperature, that.temperature) &&
               Objects.equals(timestamp, that.timestamp);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sensorId, temperature, timestamp);
    }

    @Override
    public String toString() {
        return "TemperatureData{" +
               "sensorId='" + sensorId + '\'' +
               ", temperature=" + temperature +
               ", humidity=" + humidity +
               ", location='" + location + '\'' +
               ", timestamp=" + timestamp +
               ", deviceType='" + deviceType + '\'' +
               ", batteryLevel=" + batteryLevel +
               ", signalStrength=" + signalStrength +
               '}';
    }
}
