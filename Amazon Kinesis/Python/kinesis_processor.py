import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KinesisDataProcessor:
    def __init__(self, stream_name: str, aws_region: str = 'us-east-1'):
        self.kinesis_client = boto3.client('kinesis', region_name=aws_region)
        self.stream_name = stream_name
        self.shard_iterators = {}
        
    def get_shard_iterator(self, shard_id: str, iterator_type: str = 'LATEST'):
        """Get shard iterator for reading from Kinesis stream"""
        try:
            response = self.kinesis_client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard_id,
                ShardIteratorType=iterator_type
            )
            return response['ShardIterator']
        except Exception as e:
            logger.error(f"Error getting shard iterator: {e}")
            return None
    
    def get_records(self, shard_iterator: str, limit: int = 100):
        """Get records from Kinesis stream"""
        try:
            response = self.kinesis_client.get_records(
                ShardIterator=shard_iterator,
                Limit=limit
            )
            return response['Records'], response.get('NextShardIterator')
        except Exception as e:
            logger.error(f"Error getting records: {e}")
            return [], None
    
    def process_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process Kinesis records into pandas DataFrame"""
        processed_data = []
        
        for record in records:
            try:
                # Decode the data
                data = json.loads(record['Data'])
                
                # Add Kinesis metadata
                data['kinesis_timestamp'] = record['ApproximateArrivalTimestamp']
                data['sequence_number'] = record['SequenceNumber']
                
                processed_data.append(data)
                
            except Exception as e:
                logger.error(f"Error processing record: {e}")
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['kinesis_timestamp'] = pd.to_datetime(df['kinesis_timestamp'])
            return df
        else:
            return pd.DataFrame()
    
    def aggregate_data(self, df: pd.DataFrame, window: str = '1H') -> pd.DataFrame:
        """Aggregate temperature data by time window"""
        if df.empty:
            return df
            
        # Group by sensor and time window
        aggregated = df.groupby(['sensor_id', pd.Grouper(key='timestamp', freq=window)]).agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean', 'min', 'max'],
        }).round(2)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        
        return aggregated
    
    def start_processing(self):
        """Start processing Kinesis stream"""
        # Get stream description
        try:
            response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            shards = response['StreamDescription']['Shards']
            
            logger.info(f"Found {len(shards)} shards in stream {self.stream_name}")
            
            # Initialize shard iterators
            for shard in shards:
                shard_id = shard['ShardId']
                iterator = self.get_shard_iterator(shard_id)
                if iterator:
                    self.shard_iterators[shard_id] = iterator
            
            # Process records continuously
            while True:
                for shard_id, iterator in list(self.shard_iterators.items()):
                    if iterator:
                        records, next_iterator = self.get_records(iterator)
                        
                        if records:
                            df = self.process_records(records)
                            if not df.empty:
                                logger.info(f"Processed {len(df)} records from shard {shard_id}")
                                
                                # Aggregate data
                                aggregated_df = self.aggregate_data(df)
                                if not aggregated_df.empty:
                                    logger.info(f"Aggregated data shape: {aggregated_df.shape}")
                                    
                                    # Here you can save to database or send to ML pipeline
                                    self.save_processed_data(aggregated_df)
                        
                        # Update iterator
                        self.shard_iterators[shard_id] = next_iterator
                
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to database or file"""
        # This is a placeholder - implement your preferred storage
        logger.info("Saving processed data...")
        # Example: save to CSV, database, or send to another service

if __name__ == "__main__":
    processor = KinesisDataProcessor("temperature-sensor-stream")
    processor.start_processing()