"""
AWS Glue Data Catalog Setup Script
===================================

This script sets up the AWS Glue Data Catalog for weather data ETL pipeline.
It creates databases, tables, and crawlers for data discovery and schema management.
"""

import boto3
import json
from datetime import datetime
from typing import Dict, List

class GlueDataCatalogManager:
    """
    Manages AWS Glue Data Catalog setup for weather data pipeline
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize Glue Data Catalog Manager
        
        Args:
            region_name: AWS region name
        """
        self.glue_client = boto3.client('glue', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.region = region_name
        
    def create_weather_database(self, database_name: str = "weather_analytics_db") -> bool:
        """
        Create AWS Glue database for weather data
        
        Args:
            database_name: Name of the database to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.glue_client.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': 'Weather analytics database for ETL pipeline',
                    'LocationUri': f's3://weather-data-lake/{database_name}/',
                    'Parameters': {
                        'created_by': 'weather-etl-pipeline',
                        'created_at': datetime.now().isoformat(),
                        'purpose': 'weather_data_analytics'
                    }
                }
            )
            print(f"Successfully created database: {database_name}")
            return True
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Database {database_name} already exists")
            return True
            
        except Exception as e:
            print(f"Error creating database {database_name}: {str(e)}")
            return False
    
    def create_raw_weather_table(self, database_name: str, table_name: str = "raw_weather_data") -> bool:
        """
        Create table for raw weather station data
        """
        try:
            table_input = {
                'Name': table_name,
                'Description': 'Raw weather station data from IoT devices',
                'StorageDescriptor': {
                    'Columns': [
                        {'Name': 'station_id', 'Type': 'string', 'Comment': 'Weather station identifier'},
                        {'Name': 'timestamp', 'Type': 'timestamp', 'Comment': 'Measurement timestamp'},
                        {'Name': 'temperature', 'Type': 'double', 'Comment': 'Temperature in Celsius'},
                        {'Name': 'humidity', 'Type': 'double', 'Comment': 'Humidity percentage'},
                        {'Name': 'pressure', 'Type': 'double', 'Comment': 'Atmospheric pressure in hPa'},
                        {'Name': 'wind_speed', 'Type': 'double', 'Comment': 'Wind speed in km/h'},
                        {'Name': 'wind_direction', 'Type': 'double', 'Comment': 'Wind direction in degrees'},
                        {'Name': 'weather_condition', 'Type': 'string', 'Comment': 'Weather condition description'},
                        {'Name': 'metadata', 'Type': 'string', 'Comment': 'Additional metadata in JSON format'}
                    ],
                    'Location': 's3://weather-data-lake/raw/weather_data/',
                    'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                        'Parameters': {
                            'field.delim': ',',
                            'skip.header.line.count': '1'
                        }
                    }
                },
                'PartitionKeys': [
                    {'Name': 'year', 'Type': 'string'},
                    {'Name': 'month', 'Type': 'string'},
                    {'Name': 'day', 'Type': 'string'}
                ],
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': 'csv',
                    'compressionType': 'gzip',
                    'typeOfData': 'file',
                    'created_by': 'weather-etl-pipeline'
                }
            }
            
            response = self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )
            
            print(f"Successfully created table: {table_name}")
            return True
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Table {table_name} already exists")
            return True
            
        except Exception as e:
            print(f"Error creating table {table_name}: {str(e)}")
            return False
    
    def create_processed_weather_table(self, database_name: str, table_name: str = "processed_weather_data") -> bool:
        """
        Create table for processed weather data with derived metrics
        """
        try:
            table_input = {
                'Name': table_name,
                'Description': 'Processed weather data with derived metrics and quality scores',
                'StorageDescriptor': {
                    'Columns': [
                        {'Name': 'station_id', 'Type': 'string'},
                        {'Name': 'measurement_time', 'Type': 'timestamp'},
                        {'Name': 'temperature_celsius', 'Type': 'double'},
                        {'Name': 'humidity_percent', 'Type': 'double'},
                        {'Name': 'pressure_hpa', 'Type': 'double'},
                        {'Name': 'wind_speed_kmh', 'Type': 'double'},
                        {'Name': 'wind_direction_degrees', 'Type': 'double'},
                        {'Name': 'weather_condition', 'Type': 'string'},
                        {'Name': 'heat_index_celsius', 'Type': 'double'},
                        {'Name': 'wind_chill_celsius', 'Type': 'double'},
                        {'Name': 'dew_point_celsius', 'Type': 'double'},
                        {'Name': 'processing_timestamp', 'Type': 'timestamp'},
                        {'Name': 'data_quality_score', 'Type': 'double'},
                        {'Name': 'metadata_json', 'Type': 'string'}
                    ],
                    'Location': 's3://weather-data-lake/processed/weather_data/',
                    'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    }
                },
                'PartitionKeys': [
                    {'Name': 'year', 'Type': 'string'},
                    {'Name': 'month', 'Type': 'string'},
                    {'Name': 'day', 'Type': 'string'}
                ],
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': 'parquet',
                    'compressionType': 'gzip',
                    'typeOfData': 'file'
                }
            }
            
            response = self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )
            
            print(f"Successfully created table: {table_name}")
            return True
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Table {table_name} already exists")
            return True
            
        except Exception as e:
            print(f"Error creating table {table_name}: {str(e)}")
            return False
    
    def create_weather_aggregations_table(self, database_name: str, table_name: str = "weather_aggregations") -> bool:
        """
        Create table for weather data aggregations
        """
        try:
            table_input = {
                'Name': table_name,
                'Description': 'Hourly and daily weather data aggregations by station',
                'StorageDescriptor': {
                    'Columns': [
                        {'Name': 'station_id', 'Type': 'string'},
                        {'Name': 'aggregation_level', 'Type': 'string'},
                        {'Name': 'year', 'Type': 'int'},
                        {'Name': 'month', 'Type': 'int'},
                        {'Name': 'day', 'Type': 'int'},
                        {'Name': 'hour', 'Type': 'int'},
                        {'Name': 'avg_temperature', 'Type': 'double'},
                        {'Name': 'min_temperature', 'Type': 'double'},
                        {'Name': 'max_temperature', 'Type': 'double'},
                        {'Name': 'avg_humidity', 'Type': 'double'},
                        {'Name': 'avg_pressure', 'Type': 'double'},
                        {'Name': 'avg_wind_speed', 'Type': 'double'},
                        {'Name': 'max_wind_speed', 'Type': 'double'},
                        {'Name': 'measurement_count', 'Type': 'bigint'},
                        {'Name': 'avg_data_quality', 'Type': 'double'},
                        {'Name': 'created_at', 'Type': 'timestamp'}
                    ],
                    'Location': 's3://weather-data-lake/aggregated/weather_data/',
                    'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    }
                },
                'PartitionKeys': [
                    {'Name': 'aggregation_level', 'Type': 'string'},
                    {'Name': 'year', 'Type': 'string'},
                    {'Name': 'month', 'Type': 'string'}
                ],
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': 'parquet',
                    'compressionType': 'gzip'
                }
            }
            
            response = self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )
            
            print(f"Successfully created table: {table_name}")
            return True
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Table {table_name} already exists")
            return True
            
        except Exception as e:
            print(f"Error creating table {table_name}: {str(e)}")
            return False
    
    def create_weather_crawler(self, crawler_name: str, database_name: str, s3_path: str, iam_role: str) -> bool:
        """
        Create AWS Glue Crawler for automatic schema discovery
        """
        try:
            response = self.glue_client.create_crawler(
                Name=crawler_name,
                Role=iam_role,
                DatabaseName=database_name,
                Description=f'Crawler for weather data in {s3_path}',
                Targets={
                    'S3Targets': [
                        {
                            'Path': s3_path,
                            'Exclusions': []
                        }
                    ]
                },
                Schedule='cron(0 2 * * ? *)',  # Run daily at 2 AM
                SchemaChangePolicy={
                    'UpdateBehavior': 'UPDATE_IN_DATABASE',
                    'DeleteBehavior': 'DELETE_FROM_DATABASE'
                },
                Configuration=json.dumps({
                    "Version": 1.0,
                    "CrawlerOutput": {
                        "Partitions": {
                            "AddOrUpdateBehavior": "InheritFromTable"
                        }
                    }
                })
            )
            
            print(f"Successfully created crawler: {crawler_name}")
            return True
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Crawler {crawler_name} already exists")
            return True
            
        except Exception as e:
            print(f"Error creating crawler {crawler_name}: {str(e)}")
            return False
    
    def setup_complete_data_catalog(self, database_name: str = "weather_analytics_db", iam_role: str = None) -> bool:
        """
        Setup complete AWS Glue Data Catalog for weather ETL pipeline
        """
        print("=== Setting up AWS Glue Data Catalog ===")
        
        if not iam_role:
            iam_role = "arn:aws:iam::123456789012:role/AWSGlueServiceRole"
            print(f"Using default IAM role: {iam_role}")
        
        success = True
        
        # Create database
        if not self.create_weather_database(database_name):
            success = False
        
        # Create tables
        if not self.create_raw_weather_table(database_name):
            success = False
            
        if not self.create_processed_weather_table(database_name):
            success = False
            
        if not self.create_weather_aggregations_table(database_name):
            success = False
        
        # Create crawlers
        crawlers = [
            ("weather-raw-data-crawler", "s3://weather-data-lake/raw/weather_data/"),
            ("weather-processed-data-crawler", "s3://weather-data-lake/processed/weather_data/"),
            ("weather-aggregated-data-crawler", "s3://weather-data-lake/aggregated/weather_data/")
        ]
        
        for crawler_name, s3_path in crawlers:
            if not self.create_weather_crawler(crawler_name, database_name, s3_path, iam_role):
                success = False
        
        if success:
            print("=== AWS Glue Data Catalog setup completed successfully ===")
        else:
            print("=== AWS Glue Data Catalog setup completed with some errors ===")
        
        return success

def main():
    """
    Main function to setup AWS Glue Data Catalog
    """
    # Configuration
    AWS_REGION = "us-east-1"
    DATABASE_NAME = "weather_analytics_db"
    IAM_ROLE = "arn:aws:iam::123456789012:role/AWSGlueServiceRole"  # Replace with actual role ARN
    
    # Initialize Data Catalog Manager
    catalog_manager = GlueDataCatalogManager(region_name=AWS_REGION)
    
    # Setup complete data catalog
    success = catalog_manager.setup_complete_data_catalog(
        database_name=DATABASE_NAME,
        iam_role=IAM_ROLE
    )
    
    if success:
        print("\n🎉 AWS Glue Data Catalog setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the crawlers to discover schema: aws glue start-crawler --name weather-raw-data-crawler")
        print("2. Deploy AWS Glue ETL jobs")
        print("3. Schedule ETL jobs for regular execution")
    else:
        print("\n❌ Some errors occurred during setup. Please check the logs above.")

if __name__ == "__main__":
    main()
