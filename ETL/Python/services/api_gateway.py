"""
API Gateway Service - REST API endpoints for frontend and external access
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import grpc
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from services.config import settings, logger
from proto import weather_pb2, weather_pb2_grpc


# Pydantic Models for API
class WeatherDataModel(BaseModel):
    station_id: str
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    timestamp: datetime
    metadata: Dict[str, str] = {}


class WeatherDataResponse(BaseModel):
    data: List[WeatherDataModel]
    total_count: int
    success: bool
    message: str


class StationStatsModel(BaseModel):
    station_id: str
    latest_reading: Optional[WeatherDataModel]
    avg_temperature_24h: Optional[float]
    min_temperature_24h: Optional[float]
    max_temperature_24h: Optional[float]
    data_points_24h: int


class SystemHealthModel(BaseModel):
    service: str
    healthy: bool
    status: str
    details: Dict[str, str]


class AlertModel(BaseModel):
    alert_id: str
    station_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False


# gRPC Client Manager
class GRPCClientManager:
    """Manage gRPC connections to microservices"""
    
    def __init__(self):
        self.data_storage_channel = None
        self.etl_processing_channel = None
        self.data_ingestion_channel = None
    
    async def get_data_storage_client(self):
        """Get data storage service client"""
        if not self.data_storage_channel:
            self.data_storage_channel = grpc.aio.insecure_channel(
                f"{settings.grpc_host}:{settings.grpc_port + 2}"
            )
        return weather_pb2_grpc.DataStorageServiceStub(self.data_storage_channel)
    
    async def get_etl_processing_client(self):
        """Get ETL processing service client"""
        if not self.etl_processing_channel:
            self.etl_processing_channel = grpc.aio.insecure_channel(
                f"{settings.grpc_host}:{settings.grpc_port + 1}"
            )
        return weather_pb2_grpc.ETLProcessingServiceStub(self.etl_processing_channel)
    
    async def get_data_ingestion_client(self):
        """Get data ingestion service client"""
        if not self.data_ingestion_channel:
            self.data_ingestion_channel = grpc.aio.insecure_channel(
                f"{settings.grpc_host}:{settings.grpc_port}"
            )
        return weather_pb2_grpc.DataIngestionServiceStub(self.data_ingestion_channel)
    
    async def close_connections(self):
        """Close all gRPC connections"""
        if self.data_storage_channel:
            await self.data_storage_channel.close()
        if self.etl_processing_channel:
            await self.etl_processing_channel.close()
        if self.data_ingestion_channel:
            await self.data_ingestion_channel.close()


# FastAPI Application
app = FastAPI(
    title="Weather Data ETL API",
    description="REST API for Weather Data ETL Microservices",
    version=settings.app_version
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global gRPC client manager
grpc_manager = GRPCClientManager()


def protobuf_to_model(data: weather_pb2.WeatherData) -> WeatherDataModel:
    """Convert protobuf to Pydantic model"""
    return WeatherDataModel(
        station_id=data.station_id,
        latitude=data.latitude,
        longitude=data.longitude,
        temperature=data.temperature,
        humidity=data.humidity,
        pressure=data.pressure,
        wind_speed=data.wind_speed,
        wind_direction=data.wind_direction,
        timestamp=data.timestamp.ToDatetime(),
        metadata=dict(data.metadata)
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "service": "Weather Data ETL API",
        "version": settings.app_version,
        "status": "healthy",
        "documentation": "/docs"
    }


@app.get("/health", response_model=List[SystemHealthModel])
async def health_check():
    """Check health of all microservices"""
    health_results = []
    
    try:
        # Check data storage service
        storage_client = await grpc_manager.get_data_storage_client()
        storage_health = await storage_client.HealthCheck(
            weather_pb2.HealthCheckRequest(service="data-storage")
        )
        health_results.append(SystemHealthModel(
            service="data-storage",
            healthy=storage_health.healthy,
            status=storage_health.status,
            details=dict(storage_health.details)
        ))
    except Exception as e:
        health_results.append(SystemHealthModel(
            service="data-storage",
            healthy=False,
            status=f"Connection error: {str(e)}",
            details={}
        ))
    
    try:
        # Check ETL processing service
        etl_client = await grpc_manager.get_etl_processing_client()
        etl_health = await etl_client.HealthCheck(
            weather_pb2.HealthCheckRequest(service="etl-processing")
        )
        health_results.append(SystemHealthModel(
            service="etl-processing",
            healthy=etl_health.healthy,
            status=etl_health.status,
            details=dict(etl_health.details)
        ))
    except Exception as e:
        health_results.append(SystemHealthModel(
            service="etl-processing",
            healthy=False,
            status=f"Connection error: {str(e)}",
            details={}
        ))
    
    try:
        # Check data ingestion service
        ingestion_client = await grpc_manager.get_data_ingestion_client()
        ingestion_health = await ingestion_client.HealthCheck(
            weather_pb2.HealthCheckRequest(service="data-ingestion")
        )
        health_results.append(SystemHealthModel(
            service="data-ingestion",
            healthy=ingestion_health.healthy,
            status=ingestion_health.status,
            details=dict(ingestion_health.details)
        ))
    except Exception as e:
        health_results.append(SystemHealthModel(
            service="data-ingestion",
            healthy=False,
            status=f"Connection error: {str(e)}",
            details={}
        ))
    
    return health_results


@app.get("/weather/data", response_model=WeatherDataResponse)
async def get_weather_data(
    station_id: Optional[str] = Query(None, description="Filter by station ID"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """Get weather data with optional filters"""
    try:
        storage_client = await grpc_manager.get_data_storage_client()
        
        # Create request
        request = weather_pb2.WeatherDataRequest(
            station_id=station_id or "",
            limit=limit
        )
        
        # Add timestamps if provided
        if start_time:
            request.start_time.FromDatetime(start_time)
        if end_time:
            request.end_time.FromDatetime(end_time)
        
        # Get data from storage service
        response = await storage_client.GetWeatherData(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        # Convert to API models
        weather_data = [protobuf_to_model(data) for data in response.batch.data]
        
        return WeatherDataResponse(
            data=weather_data,
            total_count=response.batch.total_count,
            success=True,
            message=response.message
        )
        
    except grpc.RpcError as e:
        logger.error("gRPC error getting weather data", error=str(e))
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    except Exception as e:
        logger.error("Error getting weather data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/stations", response_model=List[str])
async def get_station_list():
    """Get list of active weather stations"""
    try:
        # Get recent data to determine active stations
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        storage_client = await grpc_manager.get_data_storage_client()
        request = weather_pb2.WeatherDataRequest(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        response = await storage_client.GetWeatherData(request)
        
        if response.success:
            stations = list(set(data.station_id for data in response.batch.data))
            return sorted(stations)
        else:
            return []
            
    except Exception as e:
        logger.error("Error getting station list", error=str(e))
        return []


@app.get("/weather/stations/{station_id}/stats", response_model=StationStatsModel)
async def get_station_stats(station_id: str):
    """Get statistics for a specific weather station"""
    try:
        # Get last 24 hours of data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        storage_client = await grpc_manager.get_data_storage_client()
        request = weather_pb2.WeatherDataRequest(
            station_id=station_id,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        response = await storage_client.GetWeatherData(request)
        
        if not response.success:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        
        data_points = response.batch.data
        
        if not data_points:
            raise HTTPException(status_code=404, detail=f"No recent data for station {station_id}")
        
        # Calculate statistics
        temperatures = [data.temperature for data in data_points]
        
        stats = StationStatsModel(
            station_id=station_id,
            latest_reading=protobuf_to_model(data_points[0]) if data_points else None,
            avg_temperature_24h=sum(temperatures) / len(temperatures) if temperatures else None,
            min_temperature_24h=min(temperatures) if temperatures else None,
            max_temperature_24h=max(temperatures) if temperatures else None,
            data_points_24h=len(data_points)
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting station stats", station_id=station_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/alerts", response_model=List[AlertModel])
async def get_weather_alerts():
    """Get current weather alerts"""
    # This would typically check for extreme weather conditions
    # For now, return a sample alert structure
    alerts = []
    
    try:
        # Check recent data for alerts
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        
        storage_client = await grpc_manager.get_data_storage_client()
        request = weather_pb2.WeatherDataRequest(
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        response = await storage_client.GetWeatherData(request)
        
        if response.success:
            for data in response.batch.data:
                # Check for extreme temperatures
                if data.temperature > 40 or data.temperature < -20:
                    alerts.append(AlertModel(
                        alert_id=f"temp_{data.station_id}_{int(datetime.now().timestamp())}",
                        station_id=data.station_id,
                        alert_type="temperature",
                        severity="high" if abs(data.temperature) > 45 else "medium",
                        message=f"Extreme temperature: {data.temperature}°C",
                        timestamp=data.timestamp.ToDatetime()
                    ))
                
                # Check for high wind speeds
                if data.wind_speed > 80:  # km/h
                    alerts.append(AlertModel(
                        alert_id=f"wind_{data.station_id}_{int(datetime.now().timestamp())}",
                        station_id=data.station_id,
                        alert_type="wind",
                        severity="high",
                        message=f"High wind speed: {data.wind_speed} km/h",
                        timestamp=data.timestamp.ToDatetime()
                    ))
        
        return alerts
        
    except Exception as e:
        logger.error("Error getting weather alerts", error=str(e))
        return []


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    # This would return Prometheus-formatted metrics
    # For now, return basic metrics
    return {
        "weather_data_points_total": 1000,
        "active_stations": 5,
        "api_requests_total": 500,
        "system_uptime_seconds": 3600
    }


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting Weather Data ETL API Gateway", version=settings.app_version)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down Weather Data ETL API Gateway")
    await grpc_manager.close_connections()


if __name__ == "__main__":
    uvicorn.run(
        "services.api_gateway:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower()
    )
