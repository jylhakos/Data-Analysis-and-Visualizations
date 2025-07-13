"""
Data Storage Service - Database operations and caching
"""
import asyncio
import grpc
from concurrent import futures
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import json
import uuid
import boto3
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import redis

from services.config import settings, logger
from proto import weather_pb2, weather_pb2_grpc


# Database Models
Base = declarative_base()


class WeatherRecord(Base):
    """Weather data database model"""
    __tablename__ = "weather_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    station_id = Column(String(50), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    wind_direction = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    metadata = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def to_protobuf(self) -> weather_pb2.WeatherData:
        """Convert database record to protobuf"""
        from google.protobuf.timestamp_pb2 import Timestamp
        
        timestamp = Timestamp()
        timestamp.FromDatetime(self.timestamp)
        
        weather_data = weather_pb2.WeatherData(
            station_id=self.station_id,
            latitude=self.latitude,
            longitude=self.longitude,
            temperature=self.temperature,
            humidity=self.humidity,
            pressure=self.pressure,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            timestamp=timestamp
        )
        
        # Add metadata
        if self.metadata:
            metadata_dict = json.loads(self.metadata)
            for key, value in metadata_dict.items():
                weather_data.metadata[key] = str(value)
        
        return weather_data


class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def store_weather_record(self, data: weather_pb2.WeatherData) -> str:
        """Store weather data in database"""
        session = self.get_session()
        try:
            # Convert protobuf timestamp to datetime
            timestamp = data.timestamp.ToDatetime()
            
            # Prepare metadata
            metadata_dict = dict(data.metadata)
            metadata_json = json.dumps(metadata_dict) if metadata_dict else None
            
            # Create database record
            record = WeatherRecord(
                station_id=data.station_id,
                latitude=data.latitude,
                longitude=data.longitude,
                temperature=data.temperature,
                humidity=data.humidity,
                pressure=data.pressure,
                wind_speed=data.wind_speed,
                wind_direction=data.wind_direction,
                timestamp=timestamp,
                metadata=metadata_json
            )
            
            session.add(record)
            session.commit()
            
            record_id = str(record.id)
            logger.info("Weather record stored", record_id=record_id, station_id=data.station_id)
            return record_id
            
        except Exception as e:
            session.rollback()
            logger.error("Error storing weather record", error=str(e), station_id=data.station_id)
            raise
        finally:
            session.close()
    
    def get_weather_records(self, station_id: Optional[str] = None, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 100) -> List[WeatherRecord]:
        """Retrieve weather records from database"""
        session = self.get_session()
        try:
            query = session.query(WeatherRecord)
            
            if station_id:
                query = query.filter(WeatherRecord.station_id == station_id)
            
            if start_time:
                query = query.filter(WeatherRecord.timestamp >= start_time)
            
            if end_time:
                query = query.filter(WeatherRecord.timestamp <= end_time)
            
            query = query.order_by(WeatherRecord.timestamp.desc()).limit(limit)
            
            records = query.all()
            logger.info("Retrieved weather records", count=len(records))
            return records
            
        except Exception as e:
            logger.error("Error retrieving weather records", error=str(e))
            raise
        finally:
            session.close()


class CacheManager:
    """Redis cache manager for fast data access"""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        logger.info("Connected to Redis cache")
    
    def cache_weather_data(self, data: weather_pb2.WeatherData, ttl: int = 3600):
        """Cache weather data in Redis"""
        try:
            cache_key = f"weather:{data.station_id}:latest"
            
            # Convert protobuf to JSON
            cache_data = {
                "station_id": data.station_id,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "temperature": data.temperature,
                "humidity": data.humidity,
                "pressure": data.pressure,
                "wind_speed": data.wind_speed,
                "wind_direction": data.wind_direction,
                "timestamp": data.timestamp.ToDatetime().isoformat(),
                "metadata": dict(data.metadata)
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug("Weather data cached", station_id=data.station_id)
            
        except Exception as e:
            logger.error("Error caching weather data", error=str(e), station_id=data.station_id)
    
    def get_cached_weather_data(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Get cached weather data from Redis"""
        try:
            cache_key = f"weather:{station_id}:latest"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error("Error retrieving cached data", error=str(e), station_id=station_id)
            return None


class S3DataLake:
    """AWS S3 data lake for long-term storage"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=settings.aws_region)
        self.bucket_name = settings.s3_bucket_name
    
    def store_to_data_lake(self, data: weather_pb2.WeatherData):
        """Store weather data in S3 data lake"""
        try:
            # Create S3 key with date partitioning
            timestamp = data.timestamp.ToDatetime()
            s3_key = f"weather-data/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/station={data.station_id}/{uuid.uuid4()}.json"
            
            # Convert to JSON
            data_dict = {
                "station_id": data.station_id,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "temperature": data.temperature,
                "humidity": data.humidity,
                "pressure": data.pressure,
                "wind_speed": data.wind_speed,
                "wind_direction": data.wind_direction,
                "timestamp": timestamp.isoformat(),
                "metadata": dict(data.metadata)
            }
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(data_dict),
                ContentType='application/json'
            )
            
            logger.info("Data stored to S3 data lake", s3_key=s3_key, station_id=data.station_id)
            
        except Exception as e:
            logger.error("Error storing to S3 data lake", error=str(e), station_id=data.station_id)


class DataStorageServicer(weather_pb2_grpc.DataStorageServiceServicer):
    """gRPC service for data storage operations"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.s3_data_lake = S3DataLake()
    
    async def StoreWeatherData(self, request, context):
        """Store a batch of weather data"""
        try:
            stored_count = 0
            
            for data in request.batch.data:
                # Store in database
                record_id = self.db_manager.store_weather_record(data)
                
                # Cache latest data
                self.cache_manager.cache_weather_data(data)
                
                # Store in S3 data lake
                self.s3_data_lake.store_to_data_lake(data)
                
                stored_count += 1
            
            logger.info("Weather data batch stored", count=stored_count)
            
            return weather_pb2.WeatherDataResponse(
                batch=request.batch,
                success=True,
                message=f"Stored {stored_count} records"
            )
            
        except Exception as e:
            logger.error("Error storing weather data batch", error=str(e))
            return weather_pb2.WeatherDataResponse(
                batch=weather_pb2.WeatherDataBatch(),
                success=False,
                message=f"Storage error: {str(e)}"
            )
    
    async def GetWeatherData(self, request, context):
        """Retrieve weather data"""
        try:
            # Convert protobuf timestamps to datetime
            start_time = None
            end_time = None
            
            if request.start_time.seconds > 0:
                start_time = request.start_time.ToDatetime()
            
            if request.end_time.seconds > 0:
                end_time = request.end_time.ToDatetime()
            
            # Get data from database
            records = self.db_manager.get_weather_records(
                station_id=request.station_id if request.station_id else None,
                start_time=start_time,
                end_time=end_time,
                limit=request.limit if request.limit > 0 else 100
            )
            
            # Convert to protobuf
            weather_batch = weather_pb2.WeatherDataBatch()
            for record in records:
                weather_batch.data.append(record.to_protobuf())
            
            weather_batch.total_count = len(weather_batch.data)
            
            return weather_pb2.WeatherDataResponse(
                batch=weather_batch,
                success=True,
                message=f"Retrieved {len(records)} records"
            )
            
        except Exception as e:
            logger.error("Error retrieving weather data", error=str(e))
            return weather_pb2.WeatherDataResponse(
                batch=weather_pb2.WeatherDataBatch(),
                success=False,
                message=f"Retrieval error: {str(e)}"
            )
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            # Test database connection
            session = self.db_manager.get_session()
            session.execute("SELECT 1")
            session.close()
            
            # Test Redis connection
            self.cache_manager.redis_client.ping()
            
            return weather_pb2.HealthCheckResponse(
                healthy=True,
                status="Data Storage Service is healthy",
                details={"service": "data-storage", "version": settings.app_version}
            )
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return weather_pb2.HealthCheckResponse(
                healthy=False,
                status=f"Data Storage Service unhealthy: {str(e)}",
                details={"service": "data-storage", "error": str(e)}
            )


async def serve():
    """Start the data storage gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers))
    servicer = DataStorageServicer()
    
    weather_pb2_grpc.add_DataStorageServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{settings.grpc_host}:{settings.grpc_port + 2}"  # Different port
    server.add_insecure_port(listen_addr)
    
    logger.info("Starting Data Storage Service", address=listen_addr)
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Data Storage Service")
        await server.stop(0)


if __name__ == "__main__":
    asyncio.run(serve())
