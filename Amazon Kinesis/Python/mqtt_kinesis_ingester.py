import json
import boto3
import paho.mqtt.client as mqtt
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: str
    location: str

class MQTTKinesisIngester:
    def __init__(self, kinesis_stream_name: str, aws_region: str = 'us-east-1'):
        # Initialize Kinesis client
        self.kinesis_client = boto3.client('kinesis', region_name=aws_region)
        self.stream_name = kinesis_stream_name
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker successfully")
            # Subscribe to temperature sensor topics
            topics = [
                "sensors/+/temperature",  # Wildcard for multiple sensors
                "iot/temperature/+",
                "thermometer/+/data"
            ]
            for topic in topics:
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            # Parse MQTT message
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Extract sensor data
            sensor_data = self.parse_sensor_data(topic, payload)
            
            # Send to Kinesis
            self.send_to_kinesis(sensor_data)
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def parse_sensor_data(self, topic: str, payload: Dict[str, Any]) -> SensorData:
        """Parse MQTT payload into structured sensor data"""
        # Extract sensor ID from topic
        topic_parts = topic.split('/')
        sensor_id = topic_parts[1] if len(topic_parts) > 1 else "unknown"
        
        # Parse payload
        temperature = payload.get('temperature', payload.get('temp', 0.0))
        humidity = payload.get('humidity', payload.get('hum', 0.0))
        location = payload.get('location', 'unknown')
        timestamp = payload.get('timestamp', datetime.utcnow().isoformat())
        
        return SensorData(
            sensor_id=sensor_id,
            temperature=float(temperature),
            humidity=float(humidity),
            timestamp=timestamp,
            location=location
        )
    
    def send_to_kinesis(self, sensor_data: SensorData):
        """Send sensor data to Kinesis Data Stream"""
        try:
            # Prepare Kinesis record
            record = {
                'sensor_id': sensor_data.sensor_id,
                'temperature': sensor_data.temperature,
                'humidity': sensor_data.humidity,
                'timestamp': sensor_data.timestamp,
                'location': sensor_data.location
            }
            
            # Send to Kinesis
            response = self.kinesis_client.put_record(
                StreamName=self.stream_name,
                Data=json.dumps(record),
                PartitionKey=sensor_data.sensor_id
            )
            
            logger.info(f"Sent data to Kinesis: {record}")
            logger.info(f"Kinesis response: {response['ShardId']}")
            
        except Exception as e:
            logger.error(f"Error sending to Kinesis: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        logger.info("Disconnected from MQTT broker")
    
    def start(self, mqtt_host: str, mqtt_port: int = 1883):
        """Start the MQTT to Kinesis ingester"""
        try:
            self.mqtt_client.connect(mqtt_host, mqtt_port, 60)
            logger.info(f"Starting MQTT loop for broker: {mqtt_host}:{mqtt_port}")
            self.mqtt_client.loop_forever()
        except Exception as e:
            logger.error(f"Error starting MQTT client: {e}")

if __name__ == "__main__":
    # Configuration
    KINESIS_STREAM_NAME = "temperature-sensor-stream"
    MQTT_BROKER_HOST = "localhost"  # or your MQTT broker address
    AWS_REGION = "us-east-1"
    
    # Start ingester
    ingester = MQTTKinesisIngester(KINESIS_STREAM_NAME, AWS_REGION)
    ingester.start(MQTT_BROKER_HOST)