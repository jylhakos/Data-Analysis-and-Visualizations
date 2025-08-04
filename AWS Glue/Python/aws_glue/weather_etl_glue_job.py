"""
AWS Glue ETL Job for Weather Station Data Processing
=====================================

This AWS Glue script processes real-time weather station data using Spark.
It extracts data from S3, transforms it, and loads it into multiple destinations.

Based on AWS Glue programming tutorial:
https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-intro-tutorial.html
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
from datetime import datetime, timezone
import boto3

# Initialize AWS Glue context and job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Configuration
S3_BUCKET = "weather-data-lake"
DATABASE_NAME = "weather_analytics_db"
RAW_DATA_TABLE = "raw_weather_data"
PROCESSED_DATA_TABLE = "processed_weather_data"
AGGREGATED_DATA_TABLE = "weather_aggregations"

def extract_raw_weather_data():
    """
    Extract raw weather data from S3 using AWS Glue Data Catalog
    """
    print("Extracting raw weather data from S3...")
    
    # Read from Data Catalog table
    raw_weather_df = glueContext.create_dynamic_frame.from_catalog(
        database=DATABASE_NAME,
        table_name=RAW_DATA_TABLE,
        transformation_ctx="raw_weather_source"
    )
    
    print(f"Extracted {raw_weather_df.count()} records from raw weather data")
    return raw_weather_df

def validate_and_clean_data(dynamic_frame):
    """
    Validate and clean weather data using AWS Glue transforms
    """
    print("Validating and cleaning weather data...")
    
    # Convert to DataFrame for advanced transformations
    df = dynamic_frame.toDF()
    
    # Data validation rules
    validated_df = df.filter(
        (col("temperature").between(-50.0, 60.0)) &
        (col("humidity").between(0.0, 100.0)) &
        (col("pressure").between(800.0, 1200.0)) &
        (col("wind_speed") <= 200.0) &
        (col("wind_direction").between(0, 360)) &
        (col("station_id").isNotNull()) &
        (col("timestamp").isNotNull())
    )
    
    # Add data quality metrics
    total_records = df.count()
    valid_records = validated_df.count()
    data_quality_score = (valid_records / total_records) * 100 if total_records > 0 else 0
    
    print(f"Data validation complete: {valid_records}/{total_records} valid records ({data_quality_score:.2f}%)")
    
    # Convert back to DynamicFrame
    return glueContext.createDataFrame.fromDF(validated_df, glueContext, "validated_data")

def transform_weather_data(dynamic_frame):
    """
    Transform and enrich weather data with derived metrics
    """
    print("Transforming and enriching weather data...")
    
    # Apply mapping to standardize field names and types
    mapped_df = ApplyMapping.apply(
        frame=dynamic_frame,
        mappings=[
            ("station_id", "string", "station_id", "string"),
            ("timestamp", "string", "measurement_time", "timestamp"),
            ("temperature", "double", "temperature_celsius", "double"),
            ("humidity", "double", "humidity_percent", "double"),
            ("pressure", "double", "pressure_hpa", "double"),
            ("wind_speed", "double", "wind_speed_kmh", "double"),
            ("wind_direction", "double", "wind_direction_degrees", "double"),
            ("weather_condition", "string", "weather_condition", "string"),
            ("metadata", "string", "metadata_json", "string")
        ],
        transformation_ctx="apply_mapping"
    )
    
    # Convert to DataFrame for complex transformations
    df = mapped_df.toDF()
    
    # Calculate derived metrics
    transformed_df = df.withColumn(
        "heat_index_celsius",
        when(col("temperature_celsius") >= 27,
             -42.379 + 2.04901523 * col("temperature_celsius") + 
             10.14333127 * col("humidity_percent") +
             -0.22475541 * col("temperature_celsius") * col("humidity_percent")
        ).otherwise(col("temperature_celsius"))
    ).withColumn(
        "wind_chill_celsius",
        when((col("temperature_celsius") <= 10) & (col("wind_speed_kmh") >= 4.8),
             13.12 + 0.6215 * col("temperature_celsius") - 
             11.37 * pow(col("wind_speed_kmh"), 0.16) +
             0.3965 * col("temperature_celsius") * pow(col("wind_speed_kmh"), 0.16)
        ).otherwise(col("temperature_celsius"))
    ).withColumn(
        "dew_point_celsius",
        (237.7 * ((17.27 * col("temperature_celsius")) / (237.7 + col("temperature_celsius")) + 
                  log(col("humidity_percent") / 100.0))) /
        (17.27 - ((17.27 * col("temperature_celsius")) / (237.7 + col("temperature_celsius")) + 
                  log(col("humidity_percent") / 100.0)))
    ).withColumn(
        "processing_timestamp",
        current_timestamp()
    ).withColumn(
        "data_quality_score",
        lit(100.0)  # Perfect score for validated data
    ).withColumn(
        "year",
        year(col("measurement_time"))
    ).withColumn(
        "month", 
        month(col("measurement_time"))
    ).withColumn(
        "day",
        dayofmonth(col("measurement_time"))
    ).withColumn(
        "hour",
        hour(col("measurement_time"))
    )
    
    print(f"Transformation complete: {transformed_df.count()} records processed")
    
    # Convert back to DynamicFrame
    return glueContext.createDataFrame.fromDF(transformed_df, glueContext, "transformed_data")

def create_aggregations(dynamic_frame):
    """
    Create hourly and daily aggregations for analytics
    """
    print("Creating weather data aggregations...")
    
    df = dynamic_frame.toDF()
    
    # Hourly aggregations by station
    hourly_agg = df.groupBy(
        "station_id", "year", "month", "day", "hour"
    ).agg(
        avg("temperature_celsius").alias("avg_temperature"),
        min("temperature_celsius").alias("min_temperature"),
        max("temperature_celsius").alias("max_temperature"),
        avg("humidity_percent").alias("avg_humidity"),
        avg("pressure_hpa").alias("avg_pressure"),
        avg("wind_speed_kmh").alias("avg_wind_speed"),
        max("wind_speed_kmh").alias("max_wind_speed"),
        count("*").alias("measurement_count"),
        avg("data_quality_score").alias("avg_data_quality")
    ).withColumn(
        "aggregation_level",
        lit("hourly")
    ).withColumn(
        "created_at",
        current_timestamp()
    )
    
    # Daily aggregations by station
    daily_agg = df.groupBy(
        "station_id", "year", "month", "day"
    ).agg(
        avg("temperature_celsius").alias("avg_temperature"),
        min("temperature_celsius").alias("min_temperature"),
        max("temperature_celsius").alias("max_temperature"),
        avg("humidity_percent").alias("avg_humidity"),
        avg("pressure_hpa").alias("avg_pressure"),
        avg("wind_speed_kmh").alias("avg_wind_speed"),
        max("wind_speed_kmh").alias("max_wind_speed"),
        count("*").alias("measurement_count"),
        avg("data_quality_score").alias("avg_data_quality")
    ).withColumn(
        "aggregation_level",
        lit("daily")
    ).withColumn(
        "hour",
        lit(None).cast("integer")
    ).withColumn(
        "created_at",
        current_timestamp()
    )
    
    # Combine hourly and daily aggregations
    combined_agg = hourly_agg.unionAll(daily_agg)
    
    print(f"Aggregations complete: {combined_agg.count()} aggregation records created")
    
    return glueContext.createDataFrame.fromDF(combined_agg, glueContext, "aggregated_data")

def load_to_s3_and_catalog(dynamic_frame, table_name, s3_path, partition_keys=None):
    """
    Load processed data to S3 and update Data Catalog
    """
    print(f"Loading data to S3: {s3_path}")
    
    partition_keys = partition_keys or []
    
    # Write to S3 with partitioning
    glueContext.write_dynamic_frame.from_options(
        frame=dynamic_frame,
        connection_type="s3",
        format="glueparquet",
        connection_options={
            "path": s3_path,
            "partitionKeys": partition_keys
        },
        format_options={
            "compression": "gzip"
        },
        transformation_ctx=f"write_{table_name}"
    )
    
    print(f"Data successfully written to {s3_path}")

def load_to_redshift(dynamic_frame, table_name):
    """
    Load processed data to Amazon Redshift for analytics
    """
    print(f"Loading data to Redshift table: {table_name}")
    
    # Write to Redshift (requires Redshift connection in Glue)
    glueContext.write_dynamic_frame.from_options(
        frame=dynamic_frame,
        connection_type="redshift",
        connection_options={
            "redshiftTmpDir": f"s3://{S3_BUCKET}/temp/redshift/",
            "useConnectionProperties": "true",
            "dbtable": table_name,
            "connectionName": "weather-redshift-connection"
        },
        transformation_ctx=f"redshift_{table_name}"
    )
    
    print(f"Data successfully loaded to Redshift table: {table_name}")

def send_processing_metrics():
    """
    Send processing metrics to CloudWatch
    """
    cloudwatch = boto3.client('cloudwatch')
    
    # Send custom metrics
    cloudwatch.put_metric_data(
        Namespace='WeatherETL/GlueJob',
        MetricData=[
            {
                'MetricName': 'JobExecutionCount',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now(timezone.utc)
            },
            {
                'MetricName': 'ProcessingLatency',
                'Value': 0,  # Will be calculated in production
                'Unit': 'Seconds',
                'Timestamp': datetime.now(timezone.utc)
            }
        ]
    )
    
    print("Processing metrics sent to CloudWatch")

def main():
    """
    Main ETL pipeline execution
    """
    try:
        print("=== Starting AWS Glue Weather ETL Job ===")
        
        # Step 1: Extract raw data
        raw_data = extract_raw_weather_data()
        
        # Step 2: Validate and clean data
        validated_data = validate_and_clean_data(raw_data)
        
        # Step 3: Transform and enrich data
        transformed_data = transform_weather_data(validated_data)
        
        # Step 4: Create aggregations
        aggregated_data = create_aggregations(transformed_data)
        
        # Step 5: Load processed data to S3 (partitioned by year/month/day)
        load_to_s3_and_catalog(
            transformed_data,
            "processed_weather_data",
            f"s3://{S3_BUCKET}/processed/weather_data/",
            partition_keys=["year", "month", "day"]
        )
        
        # Step 6: Load aggregations to S3 (partitioned by aggregation level)
        load_to_s3_and_catalog(
            aggregated_data,
            "weather_aggregations",
            f"s3://{S3_BUCKET}/aggregated/weather_data/",
            partition_keys=["aggregation_level", "year", "month"]
        )
        
        # Step 7: Load to Redshift for analytics (optional)
        # Uncomment if Redshift connection is configured
        # load_to_redshift(transformed_data, "weather_facts")
        # load_to_redshift(aggregated_data, "weather_aggregations")
        
        # Step 8: Send processing metrics
        send_processing_metrics()
        
        print("=== AWS Glue Weather ETL Job Completed Successfully ===")
        
    except Exception as e:
        print(f"Error in ETL job: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
    
    # Commit the job
    job.commit()
