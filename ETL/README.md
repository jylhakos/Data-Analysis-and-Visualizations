# Extract, Transform, Load (ETL)

**What is ETL (Extract Transform Load)?**

Extract, transform, and load (ETL) is the process of combining data from multiple sources into a repository called a data warehouse.

**What is a data lakehouse?**

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/ETL/data_lakehouse.png?raw=true)

*Figure: A data lakehouse*

A data lakehouse is a data management system that combines the data lakes and data warehouses. At the ingestion layer, batch or streaming data arrives from a variety of sources and in a variety of formats.You convert raw format of files to Delta tables, indeed you can use the schema capabilities of Delta Lake to check for missing or unexpected data. You can use Unity Catalog to register tables according to your data governance model and required data isolation boundaries.

## How does ETL work?

Extract, transform, and load (ETL) works by moving data from the source system to the destination system at periodic intervals. 

The ETL process works in three steps.

1. Extract the relevant data from the source database.

2. Transform the data so that it is better suited for analytics.

3. Load the data into the target database.

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/ETL/extract_transform_load.png?raw=true)

*Figure: Extract, transform, and load (ETL)*

## Example of an ETL pipeline on AWS

**Extract**

Data is extracted from various sources (e.g., operational databases, application logs) using AWS Glue crawlers or Kinesis streams.

**Transform**

AWS Glue jobs clean and transform the data (e.g., data type conversion, aggregation, filtering) and store it in a structured format in Amazon S3.

**Load**

The transformed data is then loaded from S3 into Amazon Redshift for analytical querying and reporting.

## Real-time processing of temperature data using ETL with MQTT, gRPC, AWS, and a React (Next.js) dashboard

The sensor data flows through MQTT to Kafka, where dedicated ETL microservices process and store it using gRPC for efficient internal communication. 

The microservices are containerized with Docker and orchestrated with EKS and Fargate for scalable and serverless deployments. IAM roles ensure secure access to AWS resources.

Finally, a React/Next.js frontend with SSR and CloudFront delivers a fast and interactive dashboard experience to users.

1. Extract (data ingestion from external sensors to MQTT/Kafka)

Extraction microservice consumes data from the Kafka topic(s) and prepares it for transformation, perhaps using AWS Glue to simplify data transfers.

IoT Sensors

External Sensors

Temperature sensors, likely IoT devices, publish JSON data packets containing temperature readings to an MQTT broker like Mosquitto.

Deployment of temperature sensors in the relevant environments to collect data.

AWS IoT Core

Leverage AWS IoT Core as the central hub for securely ingesting and routing incoming data from the sensors.

MQTT protocol

Sensors transmit data using protocols supported by AWS IoT Core, such as MQTT.

MQTT Broker (Mosquitto)

This message broker efficiently receives and routes sensor data to subscribing services.

Kafka producer

A microservice subscribes to the relevant MQTT topic(s), extracts the JSON data, and publishes it to a Kafka topic for further processing.

2. Transform (data processing with microservices and gRPC)

Transformation consumes extracted data from Kafka, performs real-time temperature processing (e.g., unit conversion, aggregation, outlier detection using Spark Structured Streaming), and publishes the transformed data to a new Kafka topic.

AWS Kinesis data streams

Route the raw temperature data from IoT Core into Kinesis Data Streams for real-time processing and handling large volumes of streaming data.

Microservices (e.g., AWS Lambda, AWS Fargate/EKS)

The microservices to process the streamed data. 

These microservices can be deployed using serverless options like AWS Lambda or containerized solutions like AWS Fargate or Amazon EKS.

The gRPC potocol for inter-service communication

Utilize gRPC for efficient and high-performance communication between microservices, especially when dealing with real-time data streaming and frequent updates.

The gRPC's include:

HTTP/2: gRPC is built on HTTP/2, enabling features like multiplexing and server push for faster and more efficient data transfer.

Protocol buffers (Protobuf)

The gRPC uses Protobuf for data serialization, which results in compact binary messages, reducing network bandwidth usage and improving performance.

Streaming

The gRPC supports bidirectional streaming, allowing both client and server to send and receive continuous streams of messages, crucial for real-time temperature updates.

Data transformation

Within the microservices, perform transformations on the temperature data as needed, e.g., filtering, aggregation, unit conversion, anomaly detection using AWS Machine Learning services.

3. Load (data storage and visualization)

Loading service consumes the transformed data from Kafka and stores it in the chosen data stores.

Redis Cache Memory (Elastic)

Used for caching frequently accessed temperature data, improving the performance of read-heavy workloads and reducing the load on the PostgreSQL database.

RDS for PostgreSQL

RDS relational database stores the processed, structured temperature data for historical analysis, reporting, and dashboard display.

AWS Timestream

Stores the processed temperature data in a time-series database like AWS Timestream, optimized for efficient storage and analysis of time-stamped data, vital for temperature monitoring and trending.

Dashboard

The dashboard consumes processed temperature data from the PostgreSQL database (alternatively using the Redis cache) to display real-time visualizations. 

A real-time dashboard using Next.js and React to visualize the processed temperature data.

Next.js provides server-side rendering (SSR), improving initial page load times and SEO, while routing and data fetching for the React components.

Node.js is server-side runtime environment for Next.js, allowing server-side rendering logic.

Data fetching

The dashboard can fetch real-time updates from the microservices layer (which could be exposing gRPC endpoints or using other real-time communication mechanisms like WebSockets, if preferred by the frontend application).

Visualization libraries

Integrated data visualization libraries like Chart.js, D3.js, or Recharts to present temperature data in an easily digestible and insightful format using charts, graphs, and tables.

Responsive design

The dashboard adapts across different devices and incorporates interactive elements for a better user experience.

CDN (CloudFront)

Distributes the frontend assets (images, CSS, JavaScript) globally to users.

Docker

Each microservice is containerized using Docker, ensuring portability and consistent environments.

EKS (Elastic Kubernetes Service) with Fargate

EKS manages the deployment, scaling, and orchestration of the containerized microservices.

Fargate provides serverless compute capacity for the containers, eliminating the need to provision and manage servers, allowing you to pay only for the resources used by the application.

Security and access control

IAM Roles

AWS Identity and Access Management (IAM) roles define fine-grained permissions for microservices to interact with other AWS services, adhering to the principle of least privilege.

### References

What is ETL?

https://aws.amazon.com/what-is/etl/

Work with AWS Glue

https://docs.aws.amazon.com/managed-flink/latest/java/how-zeppelin-glue.html

