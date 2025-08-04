"""
AWS Glue Streaming ETL Job for Real-time Weather Data Processing
========================================================

This AWS Glue streaming job processes real-time weather data from Kinesis Data Streams
using Spark Structured Streaming with exactly-once semantics.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from awsglue import DynamicFrame
import json

# Initialize AWS Glue context and job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Configuration
KINESIS_STREAM_NAME = "weather-data-stream"
S3_CHECKPOINT_LOCATION = "s3://weather-data-lake/checkpoints/streaming/"
S3_OUTPUT_PATH = "s3://weather-data-lake/streaming/weather_data/"
WINDOW_DURATION = "10 minutes"
WATERMARK_DURATION = "2 minutes"

def process_streaming_data():
    """
    Process real-time weather data from Kinesis Data Streams
    """
    print("Starting streaming weather data processing...")
    
    # Read from Kinesis Data Streams
    kinesis_stream = glueContext.create_data_frame.from_options(
        connection_type="kinesis",
        connection_options={
            "streamName": KINESIS_STREAM_NAME,
            "startingPosition": "LATEST",
            "inferSchema": "true"
        },
        transformation_ctx="kinesis_source"
    )
    
    # Define weather data schema
    weather_schema = StructType([
        StructField("station_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("humidity", DoubleType(), True),
        StructField("pressure", DoubleType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("wind_direction", DoubleType(), True),
        StructField("weather_condition", StringType(), True)
    ])
    
    # Parse JSON data from Kinesis
    parsed_stream = kinesis_stream.select(
        from_json(col("data"), weather_schema).alias("weather_data")
    ).select("weather_data.*")
    
    # Add watermark for late data handling
    watermarked_stream = parsed_stream.withWatermark("timestamp", WATERMARK_DURATION)
    
    # Real-time data validation and enrichment
    enriched_stream = watermarked_stream.filter(
        (col("temperature").between(-50.0, 60.0)) &
        (col("humidity").between(0.0, 100.0)) &
        (col("pressure").between(800.0, 1200.0)) &
        (col("station_id").isNotNull())
    ).withColumn(
        "processing_time",
        current_timestamp()
    ).withColumn(
        "heat_index",
        when(col("temperature") >= 27,
             -42.379 + 2.04901523 * col("temperature") + 
             10.14333127 * col("humidity")
        ).otherwise(col("temperature"))
    ).withColumn(
        "alert_level",
        when(col("temperature") > 35, "HIGH_HEAT")
        .when(col("temperature") < -10, "EXTREME_COLD")
        .when(col("wind_speed") > 100, "HIGH_WIND")
        .otherwise("NORMAL")
    )
    
    return enriched_stream

def create_real_time_aggregations(stream_df):
    """
    Create real-time aggregations using windowed operations
    """
    print("Creating real-time aggregations...")
    
    # 10-minute windowed aggregations per station
    windowed_agg = stream_df.groupBy(
        window(col("timestamp"), WINDOW_DURATION),
        col("station_id")
    ).agg(
        avg("temperature").alias("avg_temperature"),
        min("temperature").alias("min_temperature"),
        max("temperature").alias("max_temperature"),
        avg("humidity").alias("avg_humidity"),
        avg("pressure").alias("avg_pressure"),
        avg("wind_speed").alias("avg_wind_speed"),
        max("wind_speed").alias("max_wind_speed"),
        count("*").alias("measurement_count"),
        collect_list("alert_level").alias("alerts")
    ).withColumn(
        "window_start",
        col("window.start")
    ).withColumn(
        "window_end",
        col("window.end")
    ).drop("window")
    
    return windowed_agg

def write_to_s3_streaming(stream_df, output_path, checkpoint_path):
    """
    Write streaming data to S3 with checkpointing
    """
    print(f"Writing streaming data to S3: {output_path}")
    
    query = stream_df.writeStream \
        .format("parquet") \
        .option("path", output_path) \
        .option("checkpointLocation", checkpoint_path) \
        .partitionBy("station_id") \
        .trigger(processingTime="30 seconds") \
        .start()
    
    return query

def main():
    """
    Main streaming ETL pipeline execution
    """
    try:
        print("=== Starting AWS Glue Streaming Weather ETL Job ===")
        
        # Process streaming data
        stream_df = process_streaming_data()
        
        # Create real-time aggregations
        aggregated_stream = create_real_time_aggregations(stream_df)
        
        # Write raw streaming data to S3
        raw_query = write_to_s3_streaming(
            stream_df,
            f"{S3_OUTPUT_PATH}/raw/",
            f"{S3_CHECKPOINT_LOCATION}/raw/"
        )
        
        # Write aggregated data to S3
        agg_query = write_to_s3_streaming(
            aggregated_stream,
            f"{S3_OUTPUT_PATH}/aggregated/",
            f"{S3_CHECKPOINT_LOCATION}/aggregated/"
        )
        
        # Wait for termination
        raw_query.awaitTermination()
        agg_query.awaitTermination()
        
        print("=== AWS Glue Streaming Weather ETL Job Completed ===")
        
    except Exception as e:
        print(f"Error in streaming ETL job: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
    
    # Commit the job
    job.commit()
