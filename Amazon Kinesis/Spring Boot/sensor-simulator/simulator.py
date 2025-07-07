#!/usr/bin/env python3
"""
IoT Temperature Sensor Simulator
Generates realistic temperature data and publishes to MQTT
"""

import json
import time
import random
import threading
from datetime import datetime
import paho.mqtt.client as mqtt
import os
import signal
import sys

class TemperatureSensor:
    def __init__(self, sensor_id, location, base_temp=20.0):
        self.sensor_id = sensor_id
        self.location = location
        self.base_temp = base_temp
        self.current_temp = base_temp
        self.humidity = random.uniform(30, 70)
        self.battery_level = random.randint(70, 100)
        self.signal_strength = random.randint(-80, -40)
        
    def get_temperature_reading(self):
        """Generate a realistic temperature reading with some variance"""
        # Add seasonal and daily patterns
        hour = datetime.now().hour
        day_of_year = datetime.now().timetuple().tm_yday
        
        # Daily temperature cycle (cooler at night, warmer during day)
        daily_variation = 5 * (0.5 + 0.5 * (1 + math.cos((hour - 14) * math.pi / 12)))
        
        # Seasonal variation (simplified)
        seasonal_variation = 10 * math.sin((day_of_year - 80) * 2 * math.pi / 365)
        
        # Random noise
        noise = random.gauss(0, 1)
        
        # Calculate temperature
        temperature = self.base_temp + daily_variation + seasonal_variation + noise
        
        # Update humidity and battery
        self.humidity += random.gauss(0, 2)
        self.humidity = max(10, min(90, self.humidity))
        
        # Battery slowly drains
        if random.random() < 0.01:  # 1% chance per reading
            self.battery_level = max(0, self.battery_level - 1)
        
        return {
            "sensor_id": self.sensor_id,
            "temperature": round(temperature, 2),
            "humidity": round(self.humidity, 2),
            "location": self.location,
            "timestamp": datetime.now().isoformat(),
            "device_type": "temperature_sensor",
            "battery_level": self.battery_level,
            "signal_strength": self.signal_strength
        }

class SensorSimulator:
    def __init__(self):
        self.mqtt_client = None
        self.sensors = []
        self.running = False
        self.threads = []
        
        # Configuration
        self.broker_url = os.getenv('MQTT_BROKER_URL', 'tcp://localhost:1883').replace('tcp://', '')
        self.broker_port = 1883
        self.sensor_count = int(os.getenv('SENSOR_COUNT', '5'))
        self.interval_seconds = int(os.getenv('INTERVAL_SECONDS', '10'))
        
        # Create sensors
        locations = ['Office', 'Kitchen', 'Living Room', 'Bedroom', 'Garage', 'Garden', 'Basement']
        base_temps = [22, 25, 21, 20, 15, 18, 16]
        
        for i in range(self.sensor_count):
            location = locations[i % len(locations)]
            base_temp = base_temps[i % len(base_temps)] + random.gauss(0, 2)
            sensor = TemperatureSensor(f"TEMP_{i+1:03d}", location, base_temp)
            self.sensors.append(sensor)
    
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client = mqtt.Client("sensor_simulator")
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_disconnect = self.on_disconnect
            
            print(f"Connecting to MQTT broker at {self.broker_url}:{self.broker_port}")
            self.mqtt_client.connect(self.broker_url, self.broker_port, 60)
            self.mqtt_client.loop_start()
            return True
            
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker successfully")
        else:
            print(f"Failed to connect to MQTT broker with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from MQTT broker")
    
    def publish_sensor_data(self, sensor):
        """Publish data from a single sensor"""
        while self.running:
            try:
                data = sensor.get_temperature_reading()
                topic = f"sensor/temperature/{sensor.sensor_id}"
                payload = json.dumps(data)
                
                if self.mqtt_client and self.mqtt_client.is_connected():
                    result = self.mqtt_client.publish(topic, payload, qos=1)
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print(f"Published: {sensor.sensor_id} = {data['temperature']}°C")
                    else:
                        print(f"Failed to publish data for {sensor.sensor_id}")
                else:
                    print("MQTT client not connected, retrying...")
                    self.connect_mqtt()
                
                time.sleep(self.interval_seconds + random.uniform(-2, 2))
                
            except Exception as e:
                print(f"Error publishing data for {sensor.sensor_id}: {e}")
                time.sleep(5)
    
    def start(self):
        """Start the sensor simulation"""
        print(f"Starting sensor simulator with {self.sensor_count} sensors")
        print(f"Publishing interval: {self.interval_seconds} seconds")
        
        if not self.connect_mqtt():
            return
        
        self.running = True
        
        # Start a thread for each sensor
        for sensor in self.sensors:
            thread = threading.Thread(target=self.publish_sensor_data, args=(sensor,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            print(f"Started thread for sensor {sensor.sensor_id} at {sensor.location}")
        
        print("All sensor threads started. Publishing data...")
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the sensor simulation"""
        print("Stopping sensor simulator...")
        self.running = False
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        print("Sensor simulator stopped")

def signal_handler(sig, frame):
    print("\nReceived interrupt signal")
    simulator.stop()
    sys.exit(0)

if __name__ == "__main__":
    import math
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("IoT Temperature Sensor Simulator")
    print("================================")
    
    simulator = SensorSimulator()
    simulator.start()
