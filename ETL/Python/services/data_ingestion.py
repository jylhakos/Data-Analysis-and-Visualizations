"""
Data Ingestion Service - MQTT client for receiving weather station data
"""
import asyncio
import json
import grpc
from concurrent import futures
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from google.protobuf.timestamp_pb2 import Timestamp

from services.config import settings, logger
from proto import weather_pb2, weather_pb2_grpc


class MQTTClient:
    """MQTT client for weather station data"""
    
    def __init__(self, data_processor):
        self.client = mqtt.Client()
        self.data_processor = data_processor
        self.setup_client()
    
    def setup_client(self):
        """Configure MQTT client"""
        self.client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection"""
        if rc == 0:
            logger.info("Connected to MQTT broker", broker=settings.mqtt_broker_host)
            # Subscribe to all weather station topics
            for station_id in range(1, settings.station_count + 1):
                topic = f"{settings.mqtt_topic_prefix}/{station_id}/data"
                client.subscribe(topic)
                logger.info("Subscribed to topic", topic=topic)
        else:
            logger.error("Failed to connect to MQTT broker", return_code=rc)
    
    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            # Parse JSON message
            data = json.loads(msg.payload.decode())
            station_id = msg.topic.split('/')[-2]
            
            logger.info("Received weather data", station_id=station_id, topic=msg.topic)
            
            # Process the data
            asyncio.create_task(self.data_processor.process_message(station_id, data))
            
        except Exception as e:
            logger.error("Error processing MQTT message", error=str(e), topic=msg.topic)
    
    def on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        logger.warning("Disconnected from MQTT broker", return_code=rc)
    
    def start(self):
        """Start MQTT client"""
        self.client.connect(settings.mqtt_broker_host, settings.mqtt_broker_port, 60)
        self.client.loop_start()
    
    def stop(self):
        """Stop MQTT client"""
        self.client.loop_stop()
        self.client.disconnect()


class DataIngestionProcessor:
    """Process and validate incoming weather data"""
    
    def __init__(self, etl_client):
        self.etl_client = etl_client
    
    async def process_message(self, station_id: str, raw_data: Dict[str, Any]):
        """Process incoming weather data message"""
        try:
            # Validate and transform data
            weather_data = self.transform_to_protobuf(station_id, raw_data)
            
            # Send to ETL service via gRPC
            request = weather_pb2.IngestDataRequest(data=weather_data)
            response = await self.etl_client.IngestWeatherData(request)
            
            if response.success:
                logger.info("Data processed successfully", 
                           station_id=station_id, record_id=response.record_id)
            else:
                logger.error("Data processing failed", 
                           station_id=station_id, message=response.message)
                
        except Exception as e:
            logger.error("Error in data processing", station_id=station_id, error=str(e))
    
    def transform_to_protobuf(self, station_id: str, data: Dict[str, Any]) -> weather_pb2.WeatherData:
        """Transform raw JSON data to protobuf format"""
        # Create timestamp
        timestamp = Timestamp()
        if 'timestamp' in data:
            dt = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        else:
            dt = datetime.now(timezone.utc)
        timestamp.FromDatetime(dt)
        
        # Create weather data protobuf
        weather_data = weather_pb2.WeatherData(
            station_id=station_id,
            latitude=float(data.get('latitude', 0.0)),
            longitude=float(data.get('longitude', 0.0)),
            temperature=float(data.get('temperature', 0.0)),
            humidity=float(data.get('humidity', 0.0)),
            pressure=float(data.get('pressure', 0.0)),
            wind_speed=float(data.get('wind_speed', 0.0)),
            wind_direction=float(data.get('wind_direction', 0.0)),
            timestamp=timestamp
        )
        
        # Add metadata
        for key, value in data.items():
            if key not in ['temperature', 'humidity', 'pressure', 'wind_speed', 
                          'wind_direction', 'latitude', 'longitude', 'timestamp']:
                weather_data.metadata[key] = str(value)
        
        return weather_data


class DataIngestionServicer(weather_pb2_grpc.DataIngestionServiceServicer):
    """gRPC service for data ingestion"""
    
    def __init__(self):
        self.processor = DataIngestionProcessor(None)  # Will be set later
    
    async def IngestWeatherData(self, request, context):
        """Handle weather data ingestion requests"""
        try:
            # Process the data
            record_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info("Processing weather data via gRPC", 
                       station_id=request.data.station_id)
            
            return weather_pb2.IngestDataResponse(
                success=True,
                message="Data ingested successfully",
                record_id=record_id
            )
            
        except Exception as e:
            logger.error("Error in gRPC ingestion", error=str(e))
            return weather_pb2.IngestDataResponse(
                success=False,
                message=f"Error: {str(e)}",
                record_id=""
            )
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        return weather_pb2.HealthCheckResponse(
            healthy=True,
            status="Data Ingestion Service is healthy",
            details={"service": "data-ingestion", "version": settings.app_version}
        )


async def serve():
    """Start the gRPC server and MQTT client"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers))
    servicer = DataIngestionServicer()
    
    weather_pb2_grpc.add_DataIngestionServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)
    
    logger.info("Starting Data Ingestion Service", address=listen_addr)
    
    # Start MQTT client
    mqtt_client = MQTTClient(servicer.processor)
    mqtt_client.start()
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Data Ingestion Service")
        mqtt_client.stop()
        await server.stop(0)


if __name__ == "__main__":
    asyncio.run(serve())
