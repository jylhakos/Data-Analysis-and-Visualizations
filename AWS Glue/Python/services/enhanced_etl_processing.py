"""
Enhanced ETL Processing Service with AWS Glue Integration
=======================================================

This service integrates the existing microservices-based ETL processing
with AWS Glue for scalable batch and streaming data processing.
"""

import asyncio
import grpc
import boto3
import json
from concurrent import futures
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from services.config import settings, logger
from proto import weather_pb2, weather_pb2_grpc
from services.etl_processing import ETLProcessor, DataValidator, DataTransformer


class AWSGlueETLManager:
    """
    Manages AWS Glue ETL jobs and data processing workflows
    """
    
    def __init__(self):
        """Initialize AWS Glue ETL Manager"""
        self.glue_client = boto3.client('glue', region_name=settings.aws_region)
        self.s3_client = boto3.client('s3', region_name=settings.aws_region)
        self.kinesis_client = boto3.client('kinesis', region_name=settings.aws_region)
        
        # Configuration
        self.batch_job_name = "weather-batch-etl-job"
        self.streaming_job_name = "weather-streaming-etl-job"
        self.kinesis_stream_name = "weather-data-stream"
        self.s3_bucket = settings.s3_bucket_name
        
    async def trigger_batch_etl_job(self, job_parameters: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Trigger AWS Glue batch ETL job for historical data processing
        
        Args:
            job_parameters: Optional job parameters
            
        Returns:
            Dict containing job run details
        """
        try:
            job_params = job_parameters or {}
            job_params.update({
                '--S3_BUCKET': self.s3_bucket,
                '--DATABASE_NAME': 'weather_analytics_db',
                '--EXECUTION_TIME': datetime.now(timezone.utc).isoformat()
            })
            
            response = self.glue_client.start_job_run(
                JobName=self.batch_job_name,
                Arguments=job_params
            )
            
            job_run_id = response['JobRunId']
            
            logger.info("AWS Glue batch ETL job triggered",
                       job_name=self.batch_job_name,
                       job_run_id=job_run_id)
            
            return {
                'success': True,
                'job_run_id': job_run_id,
                'job_name': self.batch_job_name,
                'message': 'Batch ETL job triggered successfully'
            }
            
        except Exception as e:
            logger.error("Failed to trigger batch ETL job", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to trigger batch ETL job'
            }
    
    async def start_streaming_etl_job(self) -> Dict[str, Any]:
        """
        Start AWS Glue streaming ETL job for real-time data processing
        
        Returns:
            Dict containing job run details
        """
        try:
            response = self.glue_client.start_job_run(
                JobName=self.streaming_job_name,
                Arguments={
                    '--KINESIS_STREAM_NAME': self.kinesis_stream_name,
                    '--S3_BUCKET': self.s3_bucket,
                    '--EXECUTION_TIME': datetime.now(timezone.utc).isoformat()
                }
            )
            
            job_run_id = response['JobRunId']
            
            logger.info("AWS Glue streaming ETL job started",
                       job_name=self.streaming_job_name,
                       job_run_id=job_run_id)
            
            return {
                'success': True,
                'job_run_id': job_run_id,
                'job_name': self.streaming_job_name,
                'message': 'Streaming ETL job started successfully'
            }
            
        except Exception as e:
            logger.error("Failed to start streaming ETL job", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to start streaming ETL job'
            }
    
    async def send_to_kinesis_stream(self, weather_data: weather_pb2.WeatherData) -> Dict[str, Any]:
        """
        Send weather data to Kinesis stream for real-time processing
        
        Args:
            weather_data: Weather data to send to stream
            
        Returns:
            Dict containing stream response details
        """
        try:
            # Convert protobuf to JSON
            data_dict = {
                'station_id': weather_data.station_id,
                'timestamp': weather_data.timestamp,
                'temperature': weather_data.temperature,
                'humidity': weather_data.humidity,
                'pressure': weather_data.pressure,
                'wind_speed': weather_data.wind_speed,
                'wind_direction': weather_data.wind_direction,
                'weather_condition': weather_data.weather_condition
            }
            
            # Send to Kinesis
            response = self.kinesis_client.put_record(
                StreamName=self.kinesis_stream_name,
                Data=json.dumps(data_dict),
                PartitionKey=weather_data.station_id
            )
            
            logger.info("Data sent to Kinesis stream",
                       stream_name=self.kinesis_stream_name,
                       station_id=weather_data.station_id,
                       sequence_number=response['SequenceNumber'])
            
            return {
                'success': True,
                'sequence_number': response['SequenceNumber'],
                'shard_id': response['ShardId'],
                'message': 'Data sent to Kinesis stream successfully'
            }
            
        except Exception as e:
            logger.error("Failed to send data to Kinesis stream", 
                        station_id=weather_data.station_id,
                        error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to send data to Kinesis stream'
            }
    
    async def upload_to_s3_raw(self, weather_batch: weather_pb2.WeatherDataBatch) -> Dict[str, Any]:
        """
        Upload raw weather data batch to S3 for batch processing
        
        Args:
            weather_batch: Batch of weather data to upload
            
        Returns:
            Dict containing upload details
        """
        try:
            # Convert batch to CSV format
            data_rows = []
            for data in weather_batch.data:
                row = [
                    data.station_id,
                    data.timestamp,
                    data.temperature,
                    data.humidity,
                    data.pressure,
                    data.wind_speed,
                    data.wind_direction,
                    data.weather_condition,
                    json.dumps(dict(data.metadata)) if data.metadata else '{}'
                ]
                data_rows.append(row)
            
            # Create CSV content
            df = pd.DataFrame(data_rows, columns=[
                'station_id', 'timestamp', 'temperature', 'humidity', 'pressure',
                'wind_speed', 'wind_direction', 'weather_condition', 'metadata'
            ])
            
            csv_content = df.to_csv(index=False)
            
            # Generate S3 key with partitioning
            now = datetime.now(timezone.utc)
            s3_key = f"raw/weather_data/year={now.year}/month={now.month:02d}/day={now.day:02d}/weather_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_content.encode('utf-8'),
                ContentType='text/csv'
            )
            
            logger.info("Raw weather data uploaded to S3",
                       bucket=self.s3_bucket,
                       key=s3_key,
                       record_count=len(data_rows))
            
            return {
                'success': True,
                's3_uri': f"s3://{self.s3_bucket}/{s3_key}",
                'record_count': len(data_rows),
                'message': 'Raw data uploaded to S3 successfully'
            }
            
        except Exception as e:
            logger.error("Failed to upload raw data to S3", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to upload raw data to S3'
            }
    
    async def get_job_status(self, job_run_id: str, job_name: str) -> Dict[str, Any]:
        """
        Get the status of a running AWS Glue job
        
        Args:
            job_run_id: ID of the job run to check
            job_name: Name of the job
            
        Returns:
            Dict containing job status details
        """
        try:
            response = self.glue_client.get_job_run(
                JobName=job_name,
                RunId=job_run_id
            )
            
            job_run = response['JobRun']
            
            return {
                'success': True,
                'job_run_id': job_run_id,
                'job_name': job_name,
                'job_run_state': job_run['JobRunState'],
                'started_on': job_run.get('StartedOn'),
                'completed_on': job_run.get('CompletedOn'),
                'execution_time': job_run.get('ExecutionTime'),
                'error_message': job_run.get('ErrorMessage'),
                'message': f"Job status: {job_run['JobRunState']}"
            }
            
        except Exception as e:
            logger.error("Failed to get job status", 
                        job_run_id=job_run_id,
                        job_name=job_name,
                        error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to get job status'
            }


class EnhancedETLProcessor(ETLProcessor):
    """
    Enhanced ETL processor with AWS Glue integration
    """
    
    def __init__(self):
        """Initialize enhanced ETL processor"""
        super().__init__()
        self.glue_manager = AWSGlueETLManager()
        self.batch_size_threshold = 1000  # Trigger batch processing after this many records
        self.batch_buffer = []
    
    async def process_weather_data_with_glue(self, data: weather_pb2.WeatherData, 
                                           use_streaming: bool = True) -> Dict[str, Any]:
        """
        Process weather data using both local processing and AWS Glue
        
        Args:
            data: Weather data to process
            use_streaming: Whether to send data to Kinesis stream
            
        Returns:
            Dict containing processing results
        """
        # First, do local validation and processing
        local_result = await self.process_weather_data(data)
        
        if not local_result['success']:
            return local_result
        
        # If local processing succeeded, send to AWS Glue for advanced processing
        glue_results = {}
        
        if use_streaming:
            # Send to Kinesis for real-time processing
            kinesis_result = await self.glue_manager.send_to_kinesis_stream(data)
            glue_results['kinesis'] = kinesis_result
        
        # Add to batch buffer for batch processing
        self.batch_buffer.append(data)
        
        # Check if we should trigger batch processing
        if len(self.batch_buffer) >= self.batch_size_threshold:
            batch_result = await self._process_batch_buffer()
            glue_results['batch'] = batch_result
        
        # Combine results
        combined_result = local_result.copy()
        combined_result['glue_processing'] = glue_results
        combined_result['message'] += " | AWS Glue processing initiated"
        
        return combined_result
    
    async def _process_batch_buffer(self) -> Dict[str, Any]:
        """
        Process accumulated batch data using AWS Glue
        
        Returns:
            Dict containing batch processing results
        """
        if not self.batch_buffer:
            return {'success': True, 'message': 'No data in batch buffer'}
        
        try:
            # Create batch from buffer
            batch = weather_pb2.WeatherDataBatch()
            batch.data.extend(self.batch_buffer)
            batch.total_count = len(self.batch_buffer)
            
            # Upload to S3 for batch processing
            upload_result = await self.glue_manager.upload_to_s3_raw(batch)
            
            if upload_result['success']:
                # Trigger batch ETL job
                job_result = await self.glue_manager.trigger_batch_etl_job()
                
                # Clear buffer
                processed_count = len(self.batch_buffer)
                self.batch_buffer.clear()
                
                logger.info("Batch processing completed",
                           records_processed=processed_count,
                           s3_uri=upload_result.get('s3_uri'),
                           job_run_id=job_result.get('job_run_id'))
                
                return {
                    'success': True,
                    'records_processed': processed_count,
                    'upload_result': upload_result,
                    'job_result': job_result,
                    'message': f'Batch of {processed_count} records processed'
                }
            else:
                return upload_result
                
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'message': 'Batch processing failed'
            }
    
    async def trigger_historical_processing(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Trigger historical data processing using AWS Glue batch job
        
        Args:
            start_date: Start date for historical processing (ISO format)
            end_date: End date for historical processing (ISO format)
            
        Returns:
            Dict containing job trigger results
        """
        job_parameters = {
            '--START_DATE': start_date,
            '--END_DATE': end_date,
            '--PROCESSING_TYPE': 'historical'
        }
        
        return await self.glue_manager.trigger_batch_etl_job(job_parameters)


class EnhancedETLProcessingServicer(weather_pb2_grpc.ETLProcessingServiceServicer):
    """
    Enhanced gRPC service for ETL processing with AWS Glue integration
    """
    
    def __init__(self):
        self.processor = EnhancedETLProcessor()
        self.glue_manager = AWSGlueETLManager()
    
    async def ProcessWeatherData(self, request, context):
        """Process a batch of weather data with Glue integration"""
        try:
            logger.info("Processing weather data batch with Glue integration", 
                       count=request.total_count)
            
            processed_results = []
            
            for data in request.data:
                result = await self.processor.process_weather_data_with_glue(data)
                if result['success'] and result['processed_data']:
                    processed_results.append(result['processed_data'])
            
            processed_batch = weather_pb2.WeatherDataBatch()
            processed_batch.data.extend(processed_results)
            processed_batch.total_count = len(processed_results)
            
            return weather_pb2.WeatherDataResponse(
                batch=processed_batch,
                success=True,
                message=f"Processed {len(processed_results)} records with Glue integration"
            )
            
        except Exception as e:
            logger.error("Enhanced ETL processing error", error=str(e))
            return weather_pb2.WeatherDataResponse(
                batch=weather_pb2.WeatherDataBatch(),
                success=False,
                message=f"Enhanced processing error: {str(e)}"
            )
    
    async def TriggerHistoricalProcessing(self, request, context):
        """Trigger historical data processing using AWS Glue"""
        try:
            # This would be a custom method, assuming we extend the proto definition
            start_date = request.start_date  # ISO format date string
            end_date = request.end_date      # ISO format date string
            
            result = await self.processor.trigger_historical_processing(start_date, end_date)
            
            if result['success']:
                return weather_pb2.IngestDataResponse(
                    success=True,
                    message=f"Historical processing triggered: {result['job_run_id']}",
                    record_id=result['job_run_id']
                )
            else:
                return weather_pb2.IngestDataResponse(
                    success=False,
                    message=result['message'],
                    record_id=""
                )
                
        except Exception as e:
            logger.error("Historical processing trigger error", error=str(e))
            return weather_pb2.IngestDataResponse(
                success=False,
                message=f"Failed to trigger historical processing: {str(e)}",
                record_id=""
            )
    
    async def StartStreamingETL(self, request, context):
        """Start AWS Glue streaming ETL job"""
        try:
            result = await self.glue_manager.start_streaming_etl_job()
            
            return weather_pb2.IngestDataResponse(
                success=result['success'],
                message=result['message'],
                record_id=result.get('job_run_id', '')
            )
            
        except Exception as e:
            logger.error("Streaming ETL start error", error=str(e))
            return weather_pb2.IngestDataResponse(
                success=False,
                message=f"Failed to start streaming ETL: {str(e)}",
                record_id=""
            )
    
    async def GetJobStatus(self, request, context):
        """Get status of AWS Glue job"""
        try:
            # Assuming request contains job_run_id and job_name
            job_run_id = request.record_id  # Using record_id field for job_run_id
            job_name = request.message      # Using message field for job_name
            
            result = await self.glue_manager.get_job_status(job_run_id, job_name)
            
            return weather_pb2.IngestDataResponse(
                success=result['success'],
                message=result['message'],
                record_id=job_run_id
            )
            
        except Exception as e:
            logger.error("Job status retrieval error", error=str(e))
            return weather_pb2.IngestDataResponse(
                success=False,
                message=f"Failed to get job status: {str(e)}",
                record_id=""
            )
    
    async def HealthCheck(self, request, context):
        """Enhanced health check including AWS Glue connectivity"""
        try:
            # Check Glue connectivity
            glue_healthy = True
            try:
                await self.glue_manager.glue_client.list_jobs(MaxResults=1)
            except Exception:
                glue_healthy = False
            
            health_details = {
                "service": "enhanced-etl-processing",
                "version": settings.app_version,
                "glue_connectivity": glue_healthy,
                "batch_buffer_size": len(self.processor.batch_buffer)
            }
            
            return weather_pb2.HealthCheckResponse(
                healthy=glue_healthy,
                status="Enhanced ETL Processing Service is healthy" if glue_healthy else "AWS Glue connectivity issues",
                details=health_details
            )
            
        except Exception as e:
            logger.error("Health check error", error=str(e))
            return weather_pb2.HealthCheckResponse(
                healthy=False,
                status=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )


async def serve():
    """Start the enhanced ETL processing gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers))
    servicer = EnhancedETLProcessingServicer()
    
    weather_pb2_grpc.add_ETLProcessingServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{settings.grpc_host}:{settings.grpc_port + 1}"
    server.add_insecure_port(listen_addr)
    
    logger.info("Starting Enhanced ETL Processing Service with AWS Glue integration", 
               address=listen_addr)
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Enhanced ETL Processing Service")
        await server.stop(0)


if __name__ == "__main__":
    asyncio.run(serve())
