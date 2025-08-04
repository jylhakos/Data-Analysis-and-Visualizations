"""
Shared configuration and utilities for all microservices
"""
import os
import logging
import structlog
from typing import Dict, Any
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "weather-etl-microservices"
    app_version: str = "1.0.0"
    environment: str = "development"
    
    # Database
    database_url: str = "postgresql://username:password@localhost:5432/weather_db"
    redis_url: str = "redis://localhost:6379/0"
    
    # MQTT
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_username: str = "weather_user"
    mqtt_password: str = "weather_pass"
    mqtt_topic_prefix: str = "weather/stations"
    
    # gRPC
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    grpc_max_workers: int = 10
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # AWS
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "weather-data-lake"
    
    # Weather stations
    station_count: int = 5
    collection_interval_hours: int = 1
    coverage_radius_km: int = 100
    
    # Monitoring
    log_level: str = "INFO"
    prometheus_port: int = 8080
    health_check_interval: int = 30
    
    class Config:
        env_file = ".env"


def setup_logging() -> None:
    """Configure structured logging"""
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


# Global settings instance
settings = get_settings()
setup_logging()
logger = structlog.get_logger()
