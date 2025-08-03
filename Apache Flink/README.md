# Apache Flink

Apache Flink is utilized to create different types of streaming and batch applications, including event-driven, data analytics, and data pipeline applications.

Apache Flink can be deployed on various resource providers such as Kubernetes, but also as a stand-alone cluster. 

What are event-driven applications?

An event-driven application is a stateful application that ingest events from one or more event streams and reacts to incoming events by triggering computations, state updates, or external actions.

In batch processing mode of operation to process a bounded data stream, you can choose to ingest the entire dataset before producing any results.

In stream processing involves unbounded data streams where the input may never end, and so you are forced to continuously process the data as it arrives.

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/Apache Flink/bounded-unbounded.png?raw=true)

*Figure: processing bounded and unbounded data streams*

## Data ingestion (FastAPI, Kafka and Flink)

### FastAPI

As your RESTful API framework, FastAPI handles incoming HTTP requests from the browser.

Integrating FastAPI with Kafka

Instead of directly processing the HTTP requests within FastAPI, you can use Kafka as a buffer to handle incoming events and enable asynchronous processing. 

You can write a custom Kafka producer within your FastAPI application to extract relevant data from each incoming HTTP request (e.g., request timestamp, method, path, user agent) and format this data (e.g., as JSON) and produce it to a specific Kafka topic.

### Kafka

Kafka acts as a distributed event store that collects and stores these HTTP request events.

### Flink

Flink consumption from Kafka

Apache Flink connects to the Kafka topic where your FastAPI application is publishing the HTTP request data.

Stream Processing with Flink

Flink's stream processing engine allows you to process these events in real-time. 

You can apply various operations to monitor and analyze the requests, such as filtering for identifying  specific request types or patterns or pattern detection for identifying suspicious or anomalous behavior in requests from a particular IP.

Flink can emit the processed data (e.g., monitoring metrics, alerts) to various sinks, like another Kafka topic or a database.

## Example: PyFlink event-driven application to monitor HTTP request changes in a FastAPI application

FastAPI middleware captures HTTP events in real time, streaming them through Kafka, while PyFlink processes the events as they come in.

Kafka serves as the main point for gathering and storing the raw HTTP request data from your FastAPI application.

Flink then consumes this data from Kafka and performs real-time stream processing, enabling you to build a powerful system for monitoring and analyzing HTTP requests to your FastAPI RESTful API.

This application demonstrates an event-driven architecture utilizing PyFlink for stateful stream processing. 

It effectively monitors real-time HTTP request patterns and is designed for production deployments, complete with AWS connectors for seamless integration.

Using Python DataStream API requires installing PyFlink, which is available on PyPI.

Kafka and Flink works together to build a robust real-time event processing pipeline, which can be used to monitor HTTP requests to a RESTful API built with FastAPI.

FastAPI for request logging

FastAPI Application 

Create your FastAPI application that processes incoming HTTP requests.

Request logging

Implement middleware in FastAPI to log request details (e.g., URL, timestamp) to a file or a messaging system like Kafka or Kinesis Data Streams.

### PyFlink

Execution environment

Initialize a StreamExecutionEnvironment which is the central concept for creating DataStream API programs in PyFlink.

Data source

Configure a PyFlink connector (e.g., Kinesis connector) to ingest the request logs from your messaging system.

DataStream transformations

Parsing: 

Parse the incoming request log data to extract relevant information, such as timestamps and request paths.

Keying: 

Key the data stream based on the request path to monitor changes for specific endpoints.

State management: 

Use Flink's state mechanisms, such as ValueState or ListState, to track the count of requests for each API endpoint.

Uses Flink's keyed state to track request counts per endpoint.

Windowing: 

Optionally, apply windowing to analyze request counts within specific time intervals (e.g., tumbling or sliding windows).

Implements sliding windows for metrics calculation.

Logic: 

Implement your logic within a ProcessFunction or a similar construct to monitor and react to changes in request counts.

State update: 

Update the state with the latest request count for the current time window or event.

Change detection: 

Compare the current request count with the previous state to detect significant changes or anomalies.

### AWS

Kinesis connector

Integrate the Flink Kinesis Consumer to read data from Kinesis Data Streams where your FastAPI application is logging requests.

Dependency management

Manage JAR dependencies, like the Kinesis SQL connector, required for your PyFlink application when deploying to Amazon Managed Service for Apache Flink.

Running the application

Start the services

```bash

	$ docker-compose up -d

```

Run the PyFlink processor

```bash
	
	$ python pyflink_event_processor.py

```

References

Apache Flink

https://flink.apache.org/

Use Cases

https://flink.apache.org/what-is-flink/use-cases/

Why would you use Apache Fink?

https://aws.amazon.com/what-is/apache-flink/

QuickStart: DataStream API

https://pyflink.readthedocs.io/en/main/getting_started/quickstart/datastream_api.html

DataStream API Tutorial

https://nightlies.apache.org/flink/flink-docs-release-2.0/docs/dev/python/datastream_tutorial/

Create and run a Managed Service for Apache Flink for Python application

https://docs.aws.amazon.com/managed-flink/latest/java/gs-python-createapp.html
