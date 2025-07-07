from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.descriptors import Schema, Kafka, Json, Kinesis
from pyflink.common.typeinfo import Types
import boto3

class AWSKinesisHTTPMonitor:
    """
    Example using AWS Kinesis as source/sink for HTTP monitoring
    """
    
    def __init__(self, aws_region='us-east-1'):
        self.aws_region = aws_region
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.t_env = StreamTableEnvironment.create(
            self.env,
            EnvironmentSettings.new_instance().in_streaming_mode().build()
        )
        
        # Add Kinesis connector JAR (you need to download and add this)
        self.t_env.get_config().get_configuration().set_string(
            "pipeline.jars",
            "file:///path/to/flink-connector-kinesis-1.18.0.jar"
        )
    
    def create_kinesis_source_table(self):
        """Create Kinesis source table for HTTP events"""
        
        source_ddl = f"""
        CREATE TABLE http_events_source (
            timestamp STRING,
            method STRING,
            url STRING,
            path STRING,
            client_ip STRING,
            status_code INT,
            processing_time_ms DOUBLE,
            event_type STRING,
            event_time AS TO_TIMESTAMP(timestamp),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kinesis',
            'stream' = 'http-events-stream',
            'aws.region' = '{self.aws_region}',
            'scan.stream.initpos' = 'LATEST',
            'format' = 'json'
        )
        """
        
        self.t_env.execute_sql(source_ddl)
    
    def create_kinesis_sink_table(self):
        """Create Kinesis sink table for processed metrics"""
        
        sink_ddl = f"""
        CREATE TABLE http_metrics_sink (
            timestamp STRING,
            endpoint STRING,
            request_count BIGINT,
            avg_processing_time DOUBLE,
            error_rate DOUBLE,
            window_start STRING,
            window_end STRING
        ) WITH (
            'connector' = 'kinesis',
            'stream' = 'http-metrics-stream',
            'aws.region' = '{self.aws_region}',
            'format' = 'json'
        )
        """
        
        self.t_env.execute_sql(sink_ddl)
    
    def process_http_metrics(self):
        """Process HTTP events and generate metrics"""
        
        # Windowed aggregation query
        metrics_query = """
        INSERT INTO http_metrics_sink
        SELECT
            CAST(TUMBLE_END(event_time, INTERVAL '5' MINUTE) AS STRING) as timestamp,
            path as endpoint,
            COUNT(*) as request_count,
            AVG(processing_time_ms) as avg_processing_time,
            CAST(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) * 100 as error_rate,
            CAST(TUMBLE_START(event_time, INTERVAL '5' MINUTE) AS STRING) as window_start,
            CAST(TUMBLE_END(event_time, INTERVAL '5' MINUTE) AS STRING) as window_end
        FROM http_events_source
        WHERE event_type = 'http_response'
        GROUP BY
            TUMBLE(event_time, INTERVAL '5' MINUTE),
            path
        """
        
        return self.t_env.execute_sql(metrics_query)
    
    def run(self):
        """Run the monitoring application"""
        self.create_kinesis_source_table()
        self.create_kinesis_sink_table()
        
        # Execute the processing job
        job = self.process_http_metrics()
        
        print("HTTP monitoring job started...")
        return job

# Usage example
if __name__ == "__main__":
    monitor = AWSKinesisHTTPMonitor(aws_region='us-east-1')
    job = monitor.run()