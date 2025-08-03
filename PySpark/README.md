# Apache PySpark - Air Traffic Data Analysis

PySpark is the Python API for Apache Spark.

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/PySpark/PySpark.png?raw=true)

This folder contains a Jupyter notebook (`AirTrafficProcessor.ipynb`) that demonstrates DataFrame exercises using Apache Spark to process and analyze air traffic data from 2008. The notebook includes various data analysis tasks using both Spark SQL and DataFrame API methods.

## Prerequisites

- Linux (Debian/Ubuntu) system
- Python 3.8 or higher
- Java 8 or 11 (required for Apache Spark)

## Setup

### 1. Install dependencies

First, update your package manager and install required system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv openjdk-11-jdk wget bzip2
```

### 2. Setup Java

Apache Spark requires Java to be installed and `JAVA_HOME` to be set:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

# Reload your shell configuration
source ~/.bashrc
```

Verify Java installation:
```bash
java -version
```

### 3. Create Python virtual environment

Create and activate a Python virtual environment for the project:

```bash
# Navigate to the project directory
cd "/path/to/Apache Spark"

# Create virtual environment
python3 -m venv jupyter

# Activate the virtual environment
source jupyter/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Python dependencies

Install the required Python packages:

```bash
# Install Jupyter and related packages
pip install jupyter jupyterlab

# Install data analysis libraries
pip install pyspark pandas numpy matplotlib

# Optional: Install additional useful packages
pip install plotly seaborn
```

### 5. Download and prepare data files

The notebook requires several CSV files for the exercises:

#### Download dataset

```bash
# Download the 2008 air traffic data (37MB compressed)
wget "https://dataverse.harvard.edu/api/access/datafile/1374917?gbrecs=true" -O 2008.csv.bz2

# Extract the data file
bunzip2 2008.csv.bz2
```

#### Create samples

Create smaller sample files for testing and development:

```bash
# Create sample file (1000 rows)
head -1001 2008.csv > 2008_sample.csv

# Create test sample files
head -21 2008.csv > 2008_testsample.csv
head -51 2008.csv > 2008_testsample2.csv
```

#### Create references

Create the carriers.csv file:

```bash
cat > carriers.csv << 'EOF'
Code,Description
9E,Endeavor Air Inc.
AA,American Airlines Inc.
AS,Alaska Airlines Inc.
B6,JetBlue Airways
CO,Continental Air Lines Inc.
DL,Delta Air Lines Inc.
EV,Atlantic Southeast Airlines
F9,Frontier Airlines Inc.
FL,AirTran Airways Corporation
HA,Hawaiian Airlines Inc.
MQ,American Eagle Airlines Inc.
NK,Spirit Air Lines
NW,Northwest Airlines Inc.
OH,PSA Airlines Inc.
OO,SkyWest Airlines Inc.
TZ,ATA Airlines d/b/a ATA
UA,United Air Lines Inc.
US,US Airways Inc.
WN,Southwest Airlines Co.
XE,ExpressJet Airlines Inc.
YV,Mesa Airlines Inc.
EOF
```

Create the airports.csv file:

```bash
cat > airports.csv << 'EOF'
iata,airport,city,state,country,lat,long
ATL,Hartsfield-Jackson Atlanta International,Atlanta,GA,USA,33.6407,-84.4277
DFW,Dallas/Fort Worth International,Dallas,TX,USA,32.8998,-97.0403
DEN,Denver International,Denver,CO,USA,39.8561,-104.6737
ORD,O'Hare International,Chicago,IL,USA,41.9786,-87.9048
LAX,Los Angeles International,Los Angeles,CA,USA,33.9425,-118.4081
PHX,Phoenix Sky Harbor International,Phoenix,AZ,USA,33.4343,-112.0080
LAS,McCarran International,Las Vegas,NV,USA,36.0840,-115.1537
IAH,George Bush Intercontinental,Houston,TX,USA,29.9902,-95.3368
MCO,Orlando International,Orlando,FL,USA,28.4292,-81.3089
SEA,Seattle-Tacoma International,Seattle,WA,USA,47.4502,-122.3088
MSP,Minneapolis-St Paul International,Minneapolis,MN,USA,44.8848,-93.2223
DTW,Detroit Metropolitan Wayne County,Detroit,MI,USA,42.2162,-83.3554
BOS,Logan International,Boston,MA,USA,42.3656,-71.0096
LGA,LaGuardia,New York,NY,USA,40.7769,-73.8740
JFK,John F Kennedy International,New York,NY,USA,40.6413,-73.7781
SFO,San Francisco International,San Francisco,CA,USA,37.6213,-122.3790
BWI,Baltimore/Washington International,Baltimore,MD,USA,39.1774,-76.6684
MDW,Midway International,Chicago,IL,USA,41.7868,-87.7522
PHL,Philadelphia International,Philadelphia,PA,USA,39.8744,-75.2424
DCA,Ronald Reagan Washington National,Washington,DC,USA,38.8521,-77.0377
MIA,Miami International,Miami,FL,USA,25.7959,-80.2870
SLC,Salt Lake City International,Salt Lake City,UT,USA,40.7899,-111.9791
TPA,Tampa International,Tampa,FL,USA,27.9834,-82.5330
HOU,William P Hobby,Houston,TX,USA,29.6454,-95.2789
STL,Lambert-St Louis International,St Louis,MO,USA,38.7487,-90.3700
SAN,San Diego International,San Diego,CA,USA,32.7335,-117.1896
HNL,Honolulu International,Honolulu,HI,USA,21.3099,-157.8581
PDX,Portland International,Portland,OR,USA,45.5898,-122.5951
EWR,Newark Liberty International,Newark,NJ,USA,40.6895,-74.1745
CLE,Cleveland Hopkins International,Cleveland,OH,USA,41.4117,-81.8479
EOF
```

### 6. Launch Jupyter Notebook

Start Jupyter Notebook or JupyterLab:

```bash
# Make sure virtual environment is activated
source jupyter/bin/activate

# Start Jupyter Notebook
jupyter notebook


1. **Activate the virtual environment**:
```bash
source jupyter/bin/activate
```

2. **Start Jupyter Lab**:
```bash
jupyter lab
```

3. **Open the notebook**: Navigate to `AirTrafficProcessor.ipynb`

4. **Run the first cell**: This will initialize SparkSession with MLlib capabilities

5. **Follow the exercises**: Each cell builds upon the previous ones, including both DataFrame operations and machine learning examples

Your Spark environment is now ready with full MLlib support for machine learning workflows!

## Testing the setup

### Automated verification scripts

Two verification scripts are provided to test your PySpark and MLlib installation:

#### Verification (Recommended for first-time setup):
```bash
# Make sure virtual environment is activated
source jupyter/bin/activate

# Run the quick verification script (takes ~30 seconds)
python quick_verify.py
```

Expected output:
```
PySpark & MLlib verification
========================================
🔄 Testing PySpark imports...
✅ PySpark imports successful
🔄 Testing MLlib imports...
✅ MLlib imports successful
🔄 Testing SparkSession creation...
✅ SparkSession created (Spark v3.5.x)
✅ Available cores: 8
🔄 Testing DataFrame operations...
✅ DataFrame created with 3 rows
🔄 Testing MLlib functionality...
✅ MLlib model trained (AUC: 1.000)


```

#### Verification (Detailed):
```bash
# Run the comprehensive verification script (takes 2-3 minutes)
python verify_pyspark_mllib.py
```

The comprehensive script performs the following tests:

#### **Installation tests:**
1. **Java**: Verifies Java is installed and accessible
2. **Python packages**: Checks all required packages (pyspark, pandas, numpy, matplotlib)
3. **PySpark**: Tests SparkSession creation and DataFrame operations
4. **MLlib**: Verifies all MLlib components work correctly

#### **Tests:**
5. **Air Traffic simulation**: Creates synthetic flight data and tests ML workflows
6. **Performance benchmark**: Tests processing speed with 10,000+ records

#### **Output:**
```
============================================================
  PySpark & MLlib installation verification
============================================================
🔄 Testing Java
   Java found: openjdk version "11.0.x"
✅ Java Installation
🔄 Testing Python packages
   ✅ pyspark
   ✅ pandas
   ✅ numpy
   ✅ matplotlib
✅ Python Packages
🔄 Testing PySpark
   ✅ SparkSession created successfully
   ✅ Spark version: 3.5.x
   ✅ Available cores: 8
   ✅ DataFrame operations working (count: 3)
   ✅ Spark SQL working (filtered count: 2)
✅ PySpark
🔄 Testing MLlib
   ✅ MLlib imports successful
   ✅ Test dataset created (8 rows)
   ✅ VectorAssembler working
   ✅ StringIndexer working
   ✅ StandardScaler working
   ✅ LogisticRegression training successful
   ✅ Predictions generated (8 rows)
   ✅ Model evaluation working (AUC: 1.000)
   ✅ RandomForestClassifier working
   ✅ LinearRegression working
   ✅ ML pipeline working
✅ MLlib

```

### Manual testing

You can also manually test key cells in the notebook:

1. **Cell 4** (Import and SparkSession): Should create SparkSession without errors
2. **Cell 6** (MLlib Helper Functions): Should define ML utility functions
3. **Cell 9** (MLlib Test): Should show "MLlib integration is working correctly!"
4. **Cell 12** (loadDataAndRegister): Should define the data loading function
5. **Cell 13** (Test data loading): Should display air traffic data and schema

Expected output for MLlib test (Cell 9):
```
Testing MLlib with Air Traffic data
========================================
✅ Loaded 20 rows for MLlib testing
✅ VectorAssembler test: SUCCESS
✅ Features created for 20 rows

📊 Sample features:
+-----+---------+--------+-------------------+
|Month|DayOfWeek|Distance|           features|
+-----+---------+--------+-------------------+
|    1|        3|     357|      [1.0,3.0,357.0]|
|    1|        4|     569|      [1.0,4.0,569.0]|
|    1|        4|     569|      [1.0,4.0,569.0]|
+-----+---------+--------+-------------------+

```

## Project

After setup, your directory should contain:

```
Apache Spark/
├── 2008.csv                   # Main dataset (2.3M rows)
├── 2008_sample.csv            # Sample dataset (1K rows)
├── 2008_testsample.csv        # Test dataset (20 rows)
├── 2008_testsample2.csv       # Test dataset (50 rows)
├── carriers.csv               # Airline information
├── airports.csv               # Airport information
├── AirTrafficProcessor.ipynb  # Main notebook with MLlib integration
├── README.md                  # This file
├── verify_pyspark_mllib.py    # Comprehensive verification script
├── quick_verify.py           # Quick verification script
├── test_notebook.sh          # Notebook testing script
└── jupyter/                  # Python virtual environment
    ├── bin/                  # Python executables and Spark scripts
    ├── lib/                  # Python packages (PySpark, MLlib, etc.)
    └── ...                   # Other virtual environment files
```

## What is PySpark?

**PySpark** is the Python API for Apache Spark, a unified analytics engine for large-scale data processing. It provides:

### Core Features:
- **Distributed Computing**: Process large datasets across multiple cores/machines
- **In-Memory Processing**: Much faster than traditional disk-based processing
- **SQL Interface**: Query data using familiar SQL syntax via Spark SQL
- **DataFrame API**: Similar to pandas but designed for big data
- **Fault Tolerance**: Automatic recovery from node failures
- **Lazy Evaluation**: Optimizes query execution automatically

### PySpark Components:
- **Spark Core**: Basic functionality and RDDs (Resilient Distributed Datasets)
- **Spark SQL**: SQL queries and DataFrame operations
- **Spark Streaming**: Real-time data processing
- **MLlib**: Machine learning library (covered below)
- **GraphX**: Graph processing (Scala/Java only)

## What is MLlib?

**MLlib** is Apache Spark's scalable machine learning library that provides:

### MLlib capabilities:
- **Classification**: Logistic regression, random forests, decision trees, naive Bayes
- **Regression**: Linear regression, generalized linear models, survival regression
- **Clustering**: K-means, Gaussian mixture models, hierarchical clustering
- **Collaborative Filtering**: Recommendation systems using ALS (Alternating Least Squares)
- **Dimensionality Reduction**: PCA, SVD
- **Feature Engineering**: Transformers for data preprocessing

### MLlib pipeline:
1. **Data Preparation**: Load and clean data using DataFrames
2. **Feature Engineering**: Transform data using VectorAssembler, StringIndexer, etc.
3. **Model Training**: Train ML models using algorithms like LogisticRegression
4. **Model Evaluation**: Assess performance using evaluators (AUC, RMSE, etc.)
5. **Prediction**: Apply trained models to new data

### Why MLlib for Air Traffic Data?
- **Scalability**: Handle millions of flight records efficiently
- **Integration**: Seamless with Spark SQL and DataFrames
- **Performance**: Optimized for distributed computing
- **Flexibility**: Support for batch and streaming ML workflows

## ⚙️ PySpark

The notebook is configured to run Spark in local mode with the following settings:

- **Master**: `local[*]` (uses all available CPU cores)
- **App Name**: `AirTrafficMLProcessor`
- **Dynamic Allocation**: Enabled for optimal resource usage
- **Adaptive Query Execution**: Enabled for better performance
- **Memory Settings**: 4GB driver memory, 2GB max result size

### 🔧 Configuration:
```python
spark = SparkSession.builder\
    .master("local[*]")\
    .appName("AirTrafficMLProcessor")\
    .config("spark.dynamicAllocation.enabled", "true")\
    .config("spark.shuffle.service.enabled", "true")\
    .config("spark.sql.adaptive.enabled", "true")\
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")\
    .config("spark.driver.memory", "4g")\
    .config("spark.driver.maxResultSize", "2g")\
    .getOrCreate()
```

No additional Spark installation is required as PySpark includes the necessary Spark binaries.

## Notebook

The `AirTrafficProcessor.ipynb` notebook includes exercises for:

### Data analysis exercises:
1. **Data Loading**: Loading CSV data and registering as Spark tables
2. **Flight Analysis**: Counting flights per airplane, analyzing delays
3. **Security Analysis**: Finding flights cancelled due to security
4. **Weather Analysis**: Analyzing weather-related delays
5. **Airline Analysis**: Finding airlines that didn't fly certain routes
6. **Airport Analysis**: Analyzing taxiing times and cancellation rates
7. **Statistical Analysis**: Calculating medians and percentiles
8. **Linear Regression**: Least squares analysis of delay relationships

### Machine learning with MLlib:
9. **Feature Engineering**: Using VectorAssembler and StringIndexer for ML preprocessing
10. **Delay Prediction**: Binary classification to predict flight delays
11. **Performance Analysis**: ML-based carrier and route performance evaluation
12. **Model Evaluation**: AUC, accuracy, and other ML metrics

### MLlib:
- **Classification**: LogisticRegression, RandomForestClassifier
- **Regression**: LinearRegression for continuous predictions
- **Feature Processing**: VectorAssembler, StringIndexer, StandardScaler
- **Model Evaluation**: BinaryClassificationEvaluator, RegressionEvaluator
- **Pipeline Support**: End-to-end ML workflows

## Troubleshooting

### Issues

1. **Java not found**:
   ```bash
   # Install Java
   sudo apt install openjdk-11-jdk
   
   # Set JAVA_HOME
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
   ```

2. **Permission denied**: Make sure you have write permissions in the project directory

3. **Memory issues**: For large datasets, increase JVM memory:
   ```python
   spark = SparkSession.builder \
       .master("local[*]") \
       .appName("main") \
       .config("spark.driver.memory", "4g") \
       .config("spark.driver.maxResultSize", "2g") \
       .getOrCreate()
   ```

4. **MLlib import errors**:
   ```bash
   # Reinstall PySpark
   pip uninstall pyspark
   pip install pyspark
   ```

5. **Port binding warnings**: These are normal when running multiple Spark applications:
   ```
   WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
   ```

### Virtual environment issues

If you encounter issues with the virtual environment:

```bash
# Deactivate and recreate
deactivate
rm -rf jupyter
python3 -m venv jupyter
source jupyter/bin/activate
pip install jupyter pyspark pandas numpy matplotlib
```

### Performance

For better performance with large datasets:

1. **Increase driver memory**: Add `.config("spark.driver.memory", "8g")`
2. **Optimize partitions**: Use `.coalesce()` or `.repartition()` for better data distribution
3. **Cache frequently used DataFrames**: Use `.cache()` or `.persist()`
4. **Use appropriate file formats**: Parquet is more efficient than CSV for large datasets

### Verification

Quick verification commands you can run:

```bash
# Test Java
java -version

# Test PySpark installation
python -c "import pyspark; print('PySpark version:', pyspark.__version__)"

# Test MLlib imports
python -c "from pyspark.ml.feature import VectorAssembler; print('MLlib imports successful')"

# Test SparkSession creation
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local').appName('test').getOrCreate(); print('SparkSession created successfully'); spark.stop()"
```

## Data source

The air traffic data comes from the Harvard Dataverse:
- **Source**: https://dataverse.harvard.edu/api/access/datafile/1374917
- **Description**: 2008 commercial flight data from the US Department of Transportation
- **Size**: ~234MB uncompressed, 2.3M flight records

## License

The air traffic data is public domain from the US Department of Transportation.

### References

[Program AWS Glue ETL scripts in PySpark](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python.html)