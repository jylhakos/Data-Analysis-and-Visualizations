package com.aws.etl.models;

import java.time.Instant;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;

/**
 * Represents a weather measurement from a weather station.
 * Used for both MQTT message ingestion and data lake storage.
 */
public class WeatherMeasurement {

    @JsonProperty("station_id")
    @NotNull(message = "Station ID cannot be null")
    private String stationId;

    @JsonProperty("timestamp")
    @JsonFormat(shape = JsonFormat.Shape.STRING, pattern = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", timezone = "UTC")
    @NotNull(message = "Timestamp cannot be null")
    private Instant timestamp;

    @JsonProperty("temperature")
    @DecimalMin(value = "-100.0", message = "Temperature must be greater than -100°C")
    @DecimalMax(value = "100.0", message = "Temperature must be less than 100°C")
    private Double temperature;

    @JsonProperty("humidity")
    @DecimalMin(value = "0.0", message = "Humidity must be non-negative")
    @DecimalMax(value = "100.0", message = "Humidity cannot exceed 100%")
    private Double humidity;

    @JsonProperty("pressure")
    @DecimalMin(value = "800.0", message = "Pressure must be greater than 800 hPa")
    @DecimalMax(value = "1200.0", message = "Pressure must be less than 1200 hPa")
    private Double pressure;

    @JsonProperty("wind_speed")
    @DecimalMin(value = "0.0", message = "Wind speed must be non-negative")
    private Double windSpeed;

    @JsonProperty("wind_direction")
    @DecimalMin(value = "0.0", message = "Wind direction must be non-negative")
    @DecimalMax(value = "360.0", message = "Wind direction cannot exceed 360 degrees")
    private Double windDirection;

    @JsonProperty("location")
    private StationLocation location;

    // Default constructor for Jackson
    public WeatherMeasurement() {}

    // Builder pattern constructor
    public WeatherMeasurement(Builder builder) {
        this.stationId = builder.stationId;
        this.timestamp = builder.timestamp;
        this.temperature = builder.temperature;
        this.humidity = builder.humidity;
        this.pressure = builder.pressure;
        this.windSpeed = builder.windSpeed;
        this.windDirection = builder.windDirection;
        this.location = builder.location;
    }

    // Getters and Setters
    public String getStationId() {
        return stationId;
    }

    public void setStationId(String stationId) {
        this.stationId = stationId;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Instant timestamp) {
        this.timestamp = timestamp;
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

    public Double getPressure() {
        return pressure;
    }

    public void setPressure(Double pressure) {
        this.pressure = pressure;
    }

    public Double getWindSpeed() {
        return windSpeed;
    }

    public void setWindSpeed(Double windSpeed) {
        this.windSpeed = windSpeed;
    }

    public Double getWindDirection() {
        return windDirection;
    }

    public void setWindDirection(Double windDirection) {
        this.windDirection = windDirection;
    }

    public StationLocation getLocation() {
        return location;
    }

    public void setLocation(StationLocation location) {
        this.location = location;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WeatherMeasurement that = (WeatherMeasurement) o;
        return Objects.equals(stationId, that.stationId) &&
               Objects.equals(timestamp, that.timestamp) &&
               Objects.equals(temperature, that.temperature) &&
               Objects.equals(humidity, that.humidity) &&
               Objects.equals(pressure, that.pressure) &&
               Objects.equals(windSpeed, that.windSpeed) &&
               Objects.equals(windDirection, that.windDirection) &&
               Objects.equals(location, that.location);
    }

    @Override
    public int hashCode() {
        return Objects.hash(stationId, timestamp, temperature, humidity, pressure, windSpeed, windDirection, location);
    }

    @Override
    public String toString() {
        return "WeatherMeasurement{" +
                "stationId='" + stationId + '\'' +
                ", timestamp=" + timestamp +
                ", temperature=" + temperature +
                ", humidity=" + humidity +
                ", pressure=" + pressure +
                ", windSpeed=" + windSpeed +
                ", windDirection=" + windDirection +
                ", location=" + location +
                '}';
    }

    // Builder pattern
    public static class Builder {
        private String stationId;
        private Instant timestamp;
        private Double temperature;
        private Double humidity;
        private Double pressure;
        private Double windSpeed;
        private Double windDirection;
        private StationLocation location;

        public Builder stationId(String stationId) {
            this.stationId = stationId;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder humidity(Double humidity) {
            this.humidity = humidity;
            return this;
        }

        public Builder pressure(Double pressure) {
            this.pressure = pressure;
            return this;
        }

        public Builder windSpeed(Double windSpeed) {
            this.windSpeed = windSpeed;
            return this;
        }

        public Builder windDirection(Double windDirection) {
            this.windDirection = windDirection;
            return this;
        }

        public Builder location(StationLocation location) {
            this.location = location;
            return this;
        }

        public WeatherMeasurement build() {
            return new WeatherMeasurement(this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}
