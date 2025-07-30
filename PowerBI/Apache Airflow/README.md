# Apache Airflow - Fish weight prediction pipeline

This directory contains Apache Airflow DAGs and configurations for orchestrating the Fish weight prediction machine learning pipeline with PowerBI integration.

## Overview

Apache Airflow orchestrates the entire data science workflow, from data validation to model training and PowerBI export. The pipeline ensures reproducible, automated execution of the machine learning process.

## Pipeline architecture

```mermaid
graph TD
    A[Data Validation] --> B[Data Preprocessing]
    B --> C[Model Training]
    C --> D[Generate Predictions] 
    D --> E[Export to PowerBI]
    E --> F[PowerBI Dashboard Update]
```

## 🗂️ Files Structure

```
Apache Airflow/
├── README.md                           # This file
├── fish_prediction_dag.py              # Main Airflow DAG
├── fish_supervised_pipeline.py         # Supervised learning pipeline
├── supervised_regression_pipeline.py   # Regression pipeline
├── fish_api.py                         # Fish prediction API
├── monitor_pipeline.py                 # Pipeline monitoring
├── requirements.txt                    # Airflow dependencies
├── airflow.env                         # Environment variables
├── docker-compose.fish-production.yml  # Production Docker setup
├── Dockerfile                          # Airflow Docker image
├── init-fish-db.sql                    # Database initialization
└── setup.sh                           # Airflow setup script
```

## Start

### 1. Setup Airflow environment

```bash
# Navigate to Apache Airflow directory
cd "Apache Airflow"

# Set environment variables
export AIRFLOW_HOME=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 2. Start Airflow services

```bash
# Start webserver (in background)
airflow webserver --port 8080 &

# Start scheduler (in background)  
airflow scheduler &

# Access Airflow UI at: http://localhost:8080
```

### 3. Deploy the DAG

The `fish_prediction_dag.py` is automatically detected by Airflow when placed in the DAGs folder.

## Pipeline components

### Data validation task
```python
def validate_data(**context):
    """Validates fish dataset for completeness and quality"""
    # Checks for:
    # - File existence
    # - Required columns
    # - Data quality thresholds
    # - Missing value percentages
```

**Purpose**: Ensures input data meets quality standards before processing.

### Data preprocessing task
```python
def preprocess_data(**context):
    """Preprocesses fish data for machine learning"""
    # Steps:
    # 1. Handle missing values
    # 2. Encode categorical variables
    # 3. Scale numerical features
    # 4. Create feature interactions
    # 5. Split train/test datasets
```

**Purpose**: Prepares data for optimal machine learning model performance.

### Model training task
```python
def train_model(**context):
    """Trains fish weight prediction models"""
    # Models trained:
    # - Random Forest Regressor
    # - Linear Regression
    # - Model comparison and selection
    # - Save best performing model
```

**Purpose**: Trains and validates machine learning models using scikit-learn.

### Prediction generation task
```python
def generate_predictions(**context):
    """Generates predictions for PowerBI consumption"""
    # Outputs:
    # - Individual fish weight predictions
    # - Confidence scores
    # - Model performance metrics
    # - Prediction timestamps
```

**Purpose**: Creates prediction datasets formatted for PowerBI visualization.

### PowerBI export task
```python
def export_to_powerbi(**context):
    """Exports data and results to PowerBI format"""
    # Exports:
    # - fish_data_powerbi.csv
    # - predictions_powerbi.csv  
    # - model_metrics_powerbi.csv
    # - pipeline_summary.md
```

**Purpose**: Formats and exports all data for PowerBI dashboard creation.

## 🔧 Configuration

### Environment Variables (`airflow.env`)

```bash
# Airflow Configuration
AIRFLOW_HOME=/app/Apache\ Airflow
AIRFLOW__CORE__DAGS_FOLDER=/app/Apache\ Airflow
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

# Database Configuration
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# Fish Prediction Configuration
FISH_DATA_PATH=/app/data/Fish.csv
MODEL_OUTPUT_PATH=/app/models
POWERBI_OUTPUT_PATH=/app/powerbi_output
```

### DAG configuration

```python
# DAG Settings
dag = DAG(
    'fish_weight_prediction_pipeline',
    schedule_interval=timedelta(days=1),  # Daily execution
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data-science', 'fish-prediction', 'powerbi']
)
```

## Pipeline Execution

### Manual trigger
```bash
# Trigger DAG manually
airflow dags trigger fish_weight_prediction_pipeline
```

### Scheduled execution
- **Frequency**: Daily at midnight
- **Trigger**: Automatic based on schedule_interval
- **Dependencies**: Tasks execute in sequence based on defined dependencies

### Task dependencies
```python
create_directories_task >> validate_data_task >> preprocess_data_task >> train_model_task >> generate_predictions_task >> export_to_powerbi_task
```

## PowerBI integration workflow

### 1. Data Export format
The pipeline exports three main datasets:

#### Fish data (`fish_data_powerbi.csv`)
```csv
Species,Length1,Length2,Length3,Height,Width,Weight,last_updated,data_source
Bass,23.2,25.4,30.0,11.52,4.02,270.0,2025-07-30,Fish Weight Prediction Pipeline
```

#### Predictions (`predictions_powerbi.csv`) 
```csv
species,predicted_weight,actual_weight,prediction_error,confidence_score,prediction_date
Bass,270.0,269.0,1.0,0.95,2025-07-30
```

#### Model metrics (`model_metrics_powerbi.csv`)
```csv
metric,value,model,timestamp,unit
R2_Score,0.95,Random Forest,2025-07-30,ratio
RMSE,45.2,Random Forest,2025-07-30,grams
```

### 2. PowerBI connection steps

1. **Open PowerBI Desktop**
2. **Get Data** → **Text/CSV**
3. **Import** all three CSV files from `powerbi_output/` directory
4. **Create Relationships** between datasets
5. **Build Visualizations** using the imported data

### 3. PowerBI Dashboards

#### Dashboard 1: Model performance
- **KPI Cards**: R², RMSE, MAE
- **Line Chart**: Model accuracy over time
- **Gauge**: Current model confidence

#### Dashboard 2: Fish analysis
- **Bar Chart**: Fish count by species
- **Scatter Plot**: Length vs Weight correlation
- **Histogram**: Weight distribution

#### Dashboard 3: Prediction results
- **Table**: Recent predictions with confidence
- **Scatter Plot**: Actual vs Predicted weights
- **Error Analysis**: Prediction accuracy by species

## 🐳 Docker deployment

### Production
```bash
# Build and start services
docker-compose -f docker-compose.fish-production.yml up -d

# Services include:
# - Airflow Webserver
# - Airflow Scheduler  
# - PostgreSQL Database
# - Fish Prediction API
```

### Development
```bash
# Start development environment
docker-compose up -d
```

## Monitoring and logging

### Airflow UI Monitoring
- **URL**: http://localhost:8080
- **View**: DAG runs, task status, logs, execution times
- **Alerts**: Failed task notifications

### Pipeline logs
```bash
# View Airflow logs
tail -f logs/scheduler/latest/*.log

# View DAG logs
airflow logs view fish_weight_prediction_pipeline validate_data 2025-07-30
```

### Performance metrics
- **Model Training Time**: Tracked in `model_metrics.csv`
- **Data Processing Speed**: Logged in task outputs
- **Pipeline Success Rate**: Visible in Airflow UI

## 🔧 Troubleshooting

### Issues

#### 1. DAG not appearing
```bash
# Check DAG folder
echo $AIRFLOW__CORE__DAGS_FOLDER

# Refresh DAGs
airflow dags list-import-errors
```

#### 2. Task failures
```bash
# Check task logs
airflow logs view [dag_id] [task_id] [execution_date]

# Clear failed task
airflow tasks clear [dag_id] [task_id]
```

#### 3. Database connection issues
```bash
# Test database connection
airflow db check

# Reset database
airflow db reset
```

### Performance optimization

#### 1. Parallel task execution
```python
# Configure in airflow.cfg
[celery]
worker_concurrency = 4
```

#### 2. Resource management
```python
# Set task resources
train_model_task = PythonOperator(
    task_id='train_model',
    pool='ml_pool',
    queue='ml_queue'
)
```

## Advanced features

### 1. Dynamic DAG generation
Create multiple DAGs for different fish species or models:

```python
for species in ['Bass', 'Bream', 'Pike']:
    dag_id = f'fish_prediction_{species.lower()}'
    # Generate species-specific DAG
```

### 2. External task sensors
Monitor external data sources:

```python
wait_for_data = ExternalTaskSensor(
    task_id='wait_for_data_update',
    external_dag_id='data_ingestion_dag'
)
```

### 3. Custom operators
Create reusable components:

```python
class MLModelOperator(BaseOperator):
    def execute(self, context):
        # Custom ML training logic
```

