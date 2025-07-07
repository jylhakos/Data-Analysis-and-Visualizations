from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.formats.json import JsonRowDeserializationSchema
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import ProcessFunction, KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig, Time
from pyflink.common.time import Time as CommonTime
import json
from datetime import datetime, timedelta

class HTTPRequestMonitor(KeyedProcessFunction):
    """
    Stateful stream processor for monitoring HTTP requests
    Tracks request counts per endpoint and detects anomalies
    """
    
    def __init__(self):
        self.request_count_state = None
        self.last_alert_time_state = None
        self.window_start_state = None
        
    def open(self, runtime_context):
        # State to track request count per endpoint
        request_count_descriptor = ValueStateDescriptor(
            "request_count",
            TypeInformation.of(int)
        )
        
        # State to track last alert time
        last_alert_descriptor = ValueStateDescriptor(
            "last_alert_time",
            TypeInformation.of(str)
        )
        
        # State to track window start time
        window_start_descriptor = ValueStateDescriptor(
            "window_start",
            TypeInformation.of(str)
        )
        
        # Configure state TTL (24 hours)
        ttl_config = StateTtlConfig.new_builder(CommonTime.hours(24)) \
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
            .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired) \
            .build()
        
        request_count_descriptor.enable_time_to_live(ttl_config)
        last_alert_descriptor.enable_time_to_live(ttl_config)
        window_start_descriptor.enable_time_to_live(ttl_config)
        
        self.request_count_state = runtime_context.get_state(request_count_descriptor)
        self.last_alert_time_state = runtime_context.get_state(last_alert_descriptor)
        self.window_start_state = runtime_context.get_state(window_start_descriptor)
    
    def process_element(self, value, ctx):
        current_time = ctx.timestamp()
        current_count = self.request_count_state.value()
        window_start = self.window_start_state.value()
        
        if current_count is None:
            current_count = 0
        
        # Initialize or reset window every 5 minutes
        if window_start is None or (current_time - int(window_start)) > 300000:  # 5 minutes
            self.window_start_state.update(str(current_time))
            current_count = 0
        
        # Increment request count
        current_count += 1
        self.request_count_state.update(current_count)
        
        # Parse the incoming event
        event_data = json.loads(value)
        endpoint = event_data.get('path', 'unknown')
        status_code = event_data.get('status_code', 0)
        processing_time = event_data.get('processing_time_ms', 0)
        
        # Emit metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "request_count": current_count,
            "window_duration_minutes": 5,
            "status_code": status_code,
            "processing_time_ms": processing_time,
            "event_type": "metrics"
        }
        
        yield json.dumps(metrics)
        
        # Check for anomalies (high request rate)
        if current_count > 100:  # Threshold: 100 requests per 5 minutes
            last_alert = self.last_alert_time_state.value()
            current_time_str = str(current_time)
            
            # Only alert once per hour
            if last_alert is None or (current_time - int(last_alert)) > 3600000:  # 1 hour
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "alert_type": "HIGH_REQUEST_RATE",
                    "endpoint": endpoint,
                    "request_count": current_count,
                    "threshold": 100,
                    "window_minutes": 5,
                    "severity": "WARNING"
                }
                
                self.last_alert_time_state.update(current_time_str)
                yield json.dumps(alert)
        
        # Register timer for window cleanup
        ctx.timer_service().register_processing_time_timer(current_time + 300000)  # 5 minutes
    
    def on_timer(self, timestamp, ctx):
        # Reset counters when window expires
        self.request_count_state.clear()
        self.window_start_state.clear()

class ErrorRateMonitor(KeyedProcessFunction):
    """
    Monitor error rates (4xx, 5xx status codes)
    """
    
    def __init__(self):
        self.total_requests_state = None
        self.error_requests_state = None
        self.window_start_state = None
    
    def open(self, runtime_context):
        # State descriptors
        total_descriptor = ValueStateDescriptor("total_requests", TypeInformation.of(int))
        error_descriptor = ValueStateDescriptor("error_requests", TypeInformation.of(int))
        window_descriptor = ValueStateDescriptor("window_start", TypeInformation.of(str))
        
        # Configure TTL
        ttl_config = StateTtlConfig.new_builder(CommonTime.hours(1)) \
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
            .build()
        
        total_descriptor.enable_time_to_live(ttl_config)
        error_descriptor.enable_time_to_live(ttl_config)
        window_descriptor.enable_time_to_live(ttl_config)
        
        self.total_requests_state = runtime_context.get_state(total_descriptor)
        self.error_requests_state = runtime_context.get_state(error_descriptor)
        self.window_start_state = runtime_context.get_state(window_descriptor)
    
    def process_element(self, value, ctx):
        current_time = ctx.timestamp()
        event_data = json.loads(value)
        status_code = event_data.get('status_code', 200)
        
        total_requests = self.total_requests_state.value() or 0
        error_requests = self.error_requests_state.value() or 0
        window_start = self.window_start_state.value()
        
        # Reset window every 10 minutes
        if window_start is None or (current_time - int(window_start)) > 600000:
            self.window_start_state.update(str(current_time))
            total_requests = 0
            error_requests = 0
        
        total_requests += 1
        if status_code >= 400:
            error_requests += 1
        
        self.total_requests_state.update(total_requests)
        self.error_requests_state.update(error_requests)
        
        # Calculate error rate
        error_rate = (error_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Emit error rate metrics
        error_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": event_data.get('path', 'unknown'),
            "total_requests": total_requests,
            "error_requests": error_requests,
            "error_rate_percent": round(error_rate, 2),
            "window_minutes": 10,
            "event_type": "error_metrics"
        }
        
        yield json.dumps(error_metrics)
        
        # Alert if error rate is high
        if total_requests >= 10 and error_rate > 10:  # 10% error rate with at least 10 requests
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "alert_type": "HIGH_ERROR_RATE",
                "endpoint": event_data.get('path', 'unknown'),
                "error_rate_percent": round(error_rate, 2),
                "total_requests": total_requests,
                "error_requests": error_requests,
                "severity": "CRITICAL"
            }
            yield json.dumps(alert)

def create_kafka_source(env, topic_name):
    """Create Kafka source connector"""
    kafka_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'flink-http-monitor',
        'auto.offset.reset': 'latest'
    }
    
    kafka_consumer = FlinkKafkaConsumer(
        topic_name,
        SimpleStringSchema(),
        kafka_props
    )
    
    return env.add_source(kafka_consumer)

def main():
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    # Enable checkpointing for fault tolerance
    env.enable_checkpointing(30000)  # 30 seconds
    
    # Create Kafka source for HTTP events
    http_events_stream = create_kafka_source(env, 'http-events')
    
    # Filter only HTTP response events
    response_events = http_events_stream.filter(
        lambda x: '"event_type": "http_response"' in x
    )
    
    # Key by endpoint for stateful processing
    keyed_by_endpoint = response_events.key_by(
        lambda x: json.loads(x).get('path', 'unknown')
    )
    
    # Apply request monitoring
    request_metrics = keyed_by_endpoint.process(HTTPRequestMonitor())
    
    # Apply error rate monitoring
    error_metrics = keyed_by_endpoint.process(ErrorRateMonitor())
    
    # Print results (in production, you'd send to external systems)
    request_metrics.print("REQUEST_METRICS")
    error_metrics.print("ERROR_METRICS")
    
    # Execute the job
    env.execute("HTTP Request Monitor")

if __name__ == "__main__":
    main()