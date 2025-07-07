import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemperatureSensorSimulator:
    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.sensors = {
            'sensor_001': {'location': 'Living Room', 'base_temp': 22.0},
            'sensor_002': {'location': 'Bedroom', 'base_temp': 20.0},
            'sensor_003': {'location': 'Kitchen', 'base_temp': 24.0},
            'sensor_004': {'location': 'Outdoor', 'base_temp': 15.0},
        }
        
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def generate_temperature_data(self, sensor_id: str) -> dict:
        """Generate realistic temperature data"""
        sensor_info = self.sensors[sensor_id]
        base_temp = sensor_info['base_temp']
        
        # Add daily cycle (sine wave)
        hour = datetime.now().hour
        daily_variation = 3 * np.sin(2 * np.pi * hour / 24)
        
        # Add random noise
        noise = random.gauss(0, 0.5)
        
        # Calculate temperature
        temperature = base_temp + daily_variation + noise
        
        # Generate humidity (inversely correlated with temperature)
        humidity = max(20, min(80, 60 - (temperature - 20) * 2 + random.gauss(0, 5)))
        
        return {
            'sensor_id': sensor_id,
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'location': sensor_info['location'],
            'timestamp': datetime.utcnow().isoformat(),
            'battery_level': random.randint(70, 100)
        }
    
    def publish_sensor_data(self, sensor_id: str):
        """Publish sensor data to MQTT"""
        try:
            data = self.generate_temperature_data(sensor_id)
            topic = f"sensors/{sensor_id}/temperature"
            
            self.client.publish(topic, json.dumps(data))
            logger.info(f"Published: {sensor_id} -> {data['temperature']}°C")
            
        except Exception as e:
            logger.error(f"Error publishing data for {sensor_id}: {e}")
    
    def start_simulation(self, interval: int = 10):
        """Start the sensor simulation"""
        if not self.connect():
            return
        
        logger.info(f"Starting simulation with {len(self.sensors)} sensors")
        logger.info(f"Publishing every {interval} seconds")
        
        try:
            while True:
                for sensor_id in self.sensors.keys():
                    self.publish_sensor_data(sensor_id)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.client.disconnect()

if __name__ == "__main__":
    # Configure simulation
    MQTT_BROKER = "localhost"  # Change to your MQTT broker address
    PUBLISH_INTERVAL = 30  # seconds
    
    # Start simulation
    simulator = TemperatureSensorSimulator(MQTT_BROKER)
    simulator.start_simulation(PUBLISH_INTERVAL)