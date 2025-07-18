# Data Analysis and Visualizations

Data analysis is the process of gathering, cleaning, and modeling data to reveal insights.

**Amazon Data Firehose**

Amazon Data Firehose provides way to acquire, transform, and deliver data streams to data lakes, data warehouses and analytics services.

**Amazon Kinesis**

You can use Amazon Kinesis to collect and process streams of data in real time.

**Amazon Redshift**

Amazon Redshift integrates with Amazon SageMaker, allowing you to leverage its SQL analytics capabilities. 

**Amazon SageMaker**

Analyze, prepare, and integrate data for analytics and AI.

**Apache Beam**

Apache Beam is an open source for defining both batch and streaming data-parallel processing pipelines.

You can use Apache Beam for Extract, Transform, and Load (ETL) tasks.

**Apache Flink**

Streaming Extract, Transform, and Load (ETL) with Apache Flink and Amazon Kinesis Data Analytics.

Apache Flink is a distributed processing engine for stateful computations over unbounded and bounded data streams.

**Apache Spark**

Apache Spark is an open source distributed processing system used for big data workloads.

**Google Cloud Dataflow**

Dataflow is a streaming platform provided by Google Cloud.

**Excel**

In data analytics with Excel, insights refer to the conclusions and patterns discovered after analyzing data.

**Apache Kafka**

Apache Kafka is an open-source distributed event streaming platform. The event streaming is the practice of capturing data in real-time from event sources like databases, sensors or cloud services in the form of streams of events.

**PySpark**

PySpark is the Python API for Apache Spark and tool for data analytics.

**Power BI**

Power BI is Microsoft's analytics platform that helps you turn data into actionable insights.

**Tableau**

Tableau is a data visualization tool used to analyze and present data in a visually and interactive way.

**Extract Transform and Load (ETL)**

Extract, Transform, and Load (ETL) is the process of combining data from multiple sources into a large repository called a data warehouse.

## Data sources

CSV files

Read and process CSV data

Relational database

Connect to relational database like PostgreSQL

REST API

Fetch data from REST APIs

Excel files

Handle Excel workbooks with multiple sheets


## Data analysis tasks

Data cleaning

Handle missing values, duplicates, data type conversions

Statistical analysis

Calculate means, medians, standard deviations and for example hypothesis testing for significant differences and confidence intervals and statistical significance

Conditional analysis

Identify negative values, for example calculate percentage of entries below zero

Data filtering

Filter data based on specific conditions for example to identify outliers using statistical methods

Aggregations

Group by operations and summary statistics for example to generate summary statistics by categories

Time series analysis

Date/time data handling and trends

## Leveraging Python, Pandas, NumPy, Dask, PySpark and Plotly with Tableau or Power BI for Data Visualization

Tableau or Power BI

You can integrate Python and the Plotly library with both Tableau and Power BI, allowing you to create custom visualizations and enhance data analysis capabilities.

Dash

As an alternative, consider using Plotly Dash, an open-source framework for building analytical web applications in Python.

Dash allows you to create interactive dashboards with advanced features, including data manipulation using Python libraries.

### Tableau

TabPy 

Tableau allows integration with Python through TabPy (Tableau Python Server) or the Tableau Python Data Connector (TDC). 

TabPy allows you to run Python scripts from calculated fields in Tableau workbooks.

You can embed Plotly visualizations in Tableau dashboards by the next steps:

1. Creating the visualization with Plotly.

2. Generating an embed URL using Plotly's share function.

3. Adding the URL to a data source (e.g., Google Sheets).

4. Connecting Tableau to the data source.

5. Building sheets around the Plotly visualization.

6. Adding the sheets to a dashboard and dragging a web page object to embed the URL.

```

# Custom functions for Tableau calculated fields
def detect_outliers(data):
    # Statistical outlier detection
    pass

def calculate_trend(values, dates):
    # Time series trend analysis
    pass

```


Tableau Data Connector (TDC)

Custom data source connections and data publishing

### Power BI

1. Python scripting

Enable Python scripting within Power BI Desktop to use Python visuals.

2. Add Python visuals on Power BI

Select the Python visual icon in Power BI's Visualizations pane and enable script visuals.

3. Write Python script code

In the Python script editor, write code to create your Plotly visualization. Power BI passes data to your script as a pandas DataFrame.

4. Export as an image

Since Power BI's Python visuals don't natively support interactive Plotly charts, you'll need to save the Plotly visual as an image (e.g., using kaleido) and display the image within Power BI.
