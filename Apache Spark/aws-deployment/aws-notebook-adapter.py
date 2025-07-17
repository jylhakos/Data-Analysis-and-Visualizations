# AWS Cloud Adapter for AirTrafficProcessor
# This cell should be run first when using the notebook in AWS environments

import os
import boto3
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

# Detect environment
def detect_environment():
    """Detect if running on EMR, SageMaker, or EC2"""
    if os.path.exists('/mnt/var/lib/info/'):
        return 'emr'
    elif os.path.exists('/opt/ml/'):
        return 'sagemaker'
    else:
        return 'ec2'

# AWS Environment Configuration
ENVIRONMENT = detect_environment()
print(f"🌐 Detected environment: {ENVIRONMENT.upper()}")

# Configure Spark for AWS environment
def create_aws_spark_session():
    """Create SparkSession optimized for AWS environment"""
    
    builder = SparkSession.builder.appName("AirTrafficProcessor-AWS")
    
    if ENVIRONMENT == 'emr':
        # EMR configuration
        builder = builder \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g")
    
    elif ENVIRONMENT == 'sagemaker':
        # SageMaker configuration
        builder = builder \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.maxResultSize", "1g")
    
    else:  # EC2
        # EC2 configuration
        builder = builder \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g")
    
    return builder.getOrCreate()

# S3 Data Loading Functions
def setup_s3_access():
    """Setup S3 access for data loading"""
    try:
        s3_client = boto3.client('s3')
        return s3_client
    except Exception as e:
        print(f"⚠️  S3 access setup failed: {e}")
        return None

def download_data_from_s3(bucket_name=None, local_dir="./data"):
    """Download data files from S3 to local directory"""
    
    if not bucket_name:
        # Try to auto-detect bucket name
        try:
            s3 = boto3.client('s3')
            buckets = s3.list_buckets()
            for bucket in buckets['Buckets']:
                if 'airtraffic' in bucket['Name']:
                    bucket_name = bucket['Name']
                    break
        except:
            print("❌ Could not auto-detect S3 bucket. Please specify bucket_name parameter.")
            return False
    
    if not bucket_name:
        print("❌ No S3 bucket found. Please specify bucket_name parameter.")
        return False
    
    print(f"📥 Downloading data from S3 bucket: {bucket_name}")
    
    # Create local data directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Files to download
    data_files = [
        'data/carriers.csv',
        'data/airports.csv',
        'data/2008_sample.csv',
        'data/2008_testsample.csv',
        'data/2008_testsample2.csv'
    ]
    
    # Optional large file
    large_files = ['data/2008.csv']
    
    s3 = boto3.client('s3')
    
    # Download required files
    for file_key in data_files:
        local_file = os.path.join(local_dir, os.path.basename(file_key))
        try:
            s3.download_file(bucket_name, file_key, local_file)
            print(f"✅ Downloaded: {os.path.basename(file_key)}")
        except Exception as e:
            print(f"⚠️  Failed to download {file_key}: {e}")
    
    # Download large files if available
    for file_key in large_files:
        local_file = os.path.join(local_dir, os.path.basename(file_key))
        try:
            s3.download_file(bucket_name, file_key, local_file)
            print(f"✅ Downloaded: {os.path.basename(file_key)}")
        except Exception as e:
            print(f"💡 Large file {file_key} not downloaded (optional): {e}")
    
    return True

# Enhanced loadDataAndRegister for AWS
def loadDataAndRegister_aws(path, spark_session):
    """AWS-optimized version of loadDataAndRegister"""
    
    # Check if file exists locally
    if not os.path.exists(path):
        print(f"📁 File {path} not found locally")
        
        # Try to download from S3
        bucket_name = os.environ.get('S3_BUCKET')
        if bucket_name:
            s3_path = f"s3a://{bucket_name}/data/{os.path.basename(path)}"
            print(f"🔗 Trying S3 path: {s3_path}")
            try:
                df = spark_session.read.csv(s3_path, header=True, nullValue='NA', inferSchema=True)
                df.createOrReplaceTempView("airtraffic")
                print(f"✅ Loaded data from S3: {s3_path}")
                return df
            except Exception as e:
                print(f"❌ Failed to load from S3: {e}")
        
        # Try local download
        print("🔄 Attempting to download from S3...")
        if download_data_from_s3():
            if os.path.exists(path):
                print(f"✅ File downloaded: {path}")
            else:
                print(f"❌ File still not found after download: {path}")
                return None
        else:
            print("❌ Could not download data files")
            return None
    
    # Load data locally
    try:
        df = spark_session.read.csv(path, header=True, nullValue='NA', inferSchema=True)
        df.createOrReplaceTempView("airtraffic")
        print(f"✅ Loaded data locally: {path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

# Initialize AWS environment
print("🚀 Initializing AWS environment for AirTraffic Analysis...")

# Setup S3 access
s3_client = setup_s3_access()

# Try to detect S3 bucket automatically
try:
    s3 = boto3.client('s3')
    buckets = s3.list_buckets()
    for bucket in buckets['Buckets']:
        if 'airtraffic' in bucket['Name']:
            os.environ['S3_BUCKET'] = bucket['Name']
            print(f"📊 Auto-detected S3 bucket: {bucket['Name']}")
            break
except:
    print("💡 Could not auto-detect S3 bucket. Set S3_BUCKET environment variable if needed.")

# Download data if not present
if not os.path.exists('carriers.csv'):
    print("📥 Data files not found locally. Downloading from S3...")
    download_data_from_s3()

print("✅ AWS environment setup complete!")
print("")
print("📋 Usage in AWS:")
print("1. Use create_aws_spark_session() instead of the standard SparkSession")
print("2. Use loadDataAndRegister_aws(path, spark) for data loading")
print("3. Data is automatically downloaded from S3 if not found locally")
print("")
print("🔧 Environment-specific features enabled:")
if ENVIRONMENT == 'emr':
    print("• Adaptive query execution for large datasets")
    print("• S3A connector for direct S3 access")
    print("• Optimized for distributed processing")
elif ENVIRONMENT == 'sagemaker':
    print("• SageMaker-optimized Spark configuration")
    print("• Integrated with SageMaker ML features")
    print("• Automatic S3 access with IAM roles")
else:
    print("• Local Spark cluster configuration")
    print("• Manual S3 access with boto3")
    print("• Suitable for development and testing")
