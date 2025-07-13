"""
ETL Processing Service - Data transformation and validation
"""
import asyncio
import grpc
from concurrent import futures
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from services.config import settings, logger
from proto import weather_pb2, weather_pb2_grpc


class DataValidator:
    """Validate weather data quality"""
    
    # Validation thresholds
    TEMP_MIN = -50.0  # Celsius
    TEMP_MAX = 60.0   # Celsius
    HUMIDITY_MIN = 0.0
    HUMIDITY_MAX = 100.0
    PRESSURE_MIN = 800.0  # hPa
    PRESSURE_MAX = 1200.0 # hPa
    WIND_SPEED_MAX = 200.0  # km/h
    
    @classmethod
    def validate_weather_data(cls, data: weather_pb2.WeatherData) -> Dict[str, Any]:
        """Validate weather data and return validation results"""
        errors = []
        warnings = []
        
        # Temperature validation
        if not (cls.TEMP_MIN <= data.temperature <= cls.TEMP_MAX):
            errors.append(f"Temperature {data.temperature}°C out of range [{cls.TEMP_MIN}, {cls.TEMP_MAX}]")
        
        # Humidity validation
        if not (cls.HUMIDITY_MIN <= data.humidity <= cls.HUMIDITY_MAX):
            errors.append(f"Humidity {data.humidity}% out of range [{cls.HUMIDITY_MIN}, {cls.HUMIDITY_MAX}]")
        
        # Pressure validation
        if not (cls.PRESSURE_MIN <= data.pressure <= cls.PRESSURE_MAX):
            errors.append(f"Pressure {data.pressure} hPa out of range [{cls.PRESSURE_MIN}, {cls.PRESSURE_MAX}]")
        
        # Wind speed validation
        if data.wind_speed > cls.WIND_SPEED_MAX:
            errors.append(f"Wind speed {data.wind_speed} km/h exceeds maximum {cls.WIND_SPEED_MAX}")
        
        # Wind direction validation
        if not (0 <= data.wind_direction <= 360):
            errors.append(f"Wind direction {data.wind_direction}° out of range [0, 360]")
        
        # Check for anomalous readings (could indicate sensor issues)
        if abs(data.temperature) < 0.1 and abs(data.humidity) < 0.1:
            warnings.append("Suspiciously low readings - possible sensor malfunction")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class DataTransformer:
    """Transform and enrich weather data"""
    
    @staticmethod
    def calculate_derived_metrics(data: weather_pb2.WeatherData) -> Dict[str, float]:
        """Calculate derived weather metrics"""
        # Heat index calculation (simplified)
        heat_index = DataTransformer._calculate_heat_index(data.temperature, data.humidity)
        
        # Wind chill calculation
        wind_chill = DataTransformer._calculate_wind_chill(data.temperature, data.wind_speed)
        
        # Dew point calculation
        dew_point = DataTransformer._calculate_dew_point(data.temperature, data.humidity)
        
        return {
            "heat_index": heat_index,
            "wind_chill": wind_chill,
            "dew_point": dew_point
        }
    
    @staticmethod
    def _calculate_heat_index(temp: float, humidity: float) -> float:
        """Calculate heat index (simplified formula)"""
        if temp < 27:  # Heat index only meaningful above 27°C
            return temp
        
        # Simplified heat index formula
        hi = -42.379 + 2.04901523 * temp + 10.14333127 * humidity
        hi += -0.22475541 * temp * humidity - 6.83783e-3 * temp**2
        hi += -5.481717e-2 * humidity**2 + 1.22874e-3 * temp**2 * humidity
        hi += 8.5282e-4 * temp * humidity**2 - 1.99e-6 * temp**2 * humidity**2
        
        return round(hi, 2)
    
    @staticmethod
    def _calculate_wind_chill(temp: float, wind_speed: float) -> float:
        """Calculate wind chill"""
        if temp > 10 or wind_speed < 4.8:  # Wind chill only meaningful in cold, windy conditions
            return temp
        
        # Wind chill formula (metric)
        wc = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        return round(wc, 2)
    
    @staticmethod
    def _calculate_dew_point(temp: float, humidity: float) -> float:
        """Calculate dew point using Magnus formula"""
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return round(dew_point, 2)


class ETLProcessor:
    """Main ETL processing logic"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.transformer = DataTransformer()
    
    async def process_weather_data(self, data: weather_pb2.WeatherData) -> Dict[str, Any]:
        """Process single weather data record"""
        result = {
            "original_data": data,
            "processed_data": None,
            "validation_result": None,
            "derived_metrics": None,
            "success": False,
            "message": ""
        }
        
        try:
            # Step 1: Validate data
            validation_result = self.validator.validate_weather_data(data)
            result["validation_result"] = validation_result
            
            if not validation_result["is_valid"]:
                result["message"] = f"Validation failed: {', '.join(validation_result['errors'])}"
                return result
            
            # Step 2: Transform and enrich data
            derived_metrics = self.transformer.calculate_derived_metrics(data)
            result["derived_metrics"] = derived_metrics
            
            # Step 3: Create processed data record
            processed_data = weather_pb2.WeatherData()
            processed_data.CopyFrom(data)
            
            # Add derived metrics to metadata
            for key, value in derived_metrics.items():
                processed_data.metadata[key] = str(value)
            
            # Add processing timestamp
            processed_data.metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
            processed_data.metadata["processor_version"] = settings.app_version
            
            result["processed_data"] = processed_data
            result["success"] = True
            result["message"] = "Data processed successfully"
            
            logger.info("Weather data processed", 
                       station_id=data.station_id,
                       validation_warnings=len(validation_result.get("warnings", [])))
            
        except Exception as e:
            result["message"] = f"Processing error: {str(e)}"
            logger.error("ETL processing error", station_id=data.station_id, error=str(e))
        
        return result
    
    async def process_batch(self, batch: weather_pb2.WeatherDataBatch) -> weather_pb2.WeatherDataBatch:
        """Process a batch of weather data"""
        processed_batch = weather_pb2.WeatherDataBatch()
        
        for data in batch.data:
            result = await self.process_weather_data(data)
            if result["success"] and result["processed_data"]:
                processed_batch.data.append(result["processed_data"])
        
        processed_batch.total_count = len(processed_batch.data)
        return processed_batch


class ETLProcessingServicer(weather_pb2_grpc.ETLProcessingServiceServicer):
    """gRPC service for ETL processing"""
    
    def __init__(self):
        self.processor = ETLProcessor()
    
    async def ProcessWeatherData(self, request, context):
        """Process a batch of weather data"""
        try:
            logger.info("Processing weather data batch", count=request.total_count)
            
            processed_batch = await self.processor.process_batch(request)
            
            return weather_pb2.WeatherDataResponse(
                batch=processed_batch,
                success=True,
                message=f"Processed {processed_batch.total_count} records"
            )
            
        except Exception as e:
            logger.error("ETL processing error", error=str(e))
            return weather_pb2.WeatherDataResponse(
                batch=weather_pb2.WeatherDataBatch(),
                success=False,
                message=f"Processing error: {str(e)}"
            )
    
    async def ValidateData(self, request, context):
        """Validate single weather data record"""
        try:
            validation_result = self.processor.validator.validate_weather_data(request)
            
            return weather_pb2.IngestDataResponse(
                success=validation_result["is_valid"],
                message=f"Validation: {len(validation_result['errors'])} errors, {len(validation_result['warnings'])} warnings",
                record_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        except Exception as e:
            logger.error("Validation error", error=str(e))
            return weather_pb2.IngestDataResponse(
                success=False,
                message=f"Validation error: {str(e)}",
                record_id=""
            )
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        return weather_pb2.HealthCheckResponse(
            healthy=True,
            status="ETL Processing Service is healthy",
            details={"service": "etl-processing", "version": settings.app_version}
        )


async def serve():
    """Start the ETL processing gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers))
    servicer = ETLProcessingServicer()
    
    weather_pb2_grpc.add_ETLProcessingServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{settings.grpc_host}:{settings.grpc_port + 1}"  # Different port
    server.add_insecure_port(listen_addr)
    
    logger.info("Starting ETL Processing Service", address=listen_addr)
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down ETL Processing Service")
        await server.stop(0)


if __name__ == "__main__":
    asyncio.run(serve())
