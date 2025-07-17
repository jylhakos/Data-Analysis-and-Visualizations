#!/usr/bin/env python3
"""
PySpark and MLlib Installation Verification Script
==================================================

This script verifies that PySpark and MLlib are correctly installed and configured
on your Linux system. It runs comprehensive tests to ensure all components work properly.

Usage:
    python verify_pyspark_mllib.py

Requirements:
    - Python 3.8+
    - Java 8 or 11
    - PySpark installed
    - Sample data files (optional, script creates test data if missing)
"""

import sys
import os
import time
from datetime import datetime

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_test(test_name, status="RUNNING"):
    """Print test status"""
    status_symbols = {
        "RUNNING": "🔄",
        "PASS": "✅",
        "FAIL": "❌",
        "WARNING": "⚠️"
    }
    symbol = status_symbols.get(status, "📋")
    print(f"{symbol} {test_name}")

def test_java_installation():
    """Test Java installation and JAVA_HOME"""
    print_test("Testing Java Installation", "RUNNING")
    
    try:
        import subprocess
        
        # Check Java version
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            java_version = result.stderr.split('\n')[0]
            print(f"   Java found: {java_version}")
            print_test("Java Installation", "PASS")
            return True
        else:
            print("   Java not found in PATH")
            print_test("Java Installation", "FAIL")
            return False
            
    except FileNotFoundError:
        print("   Java executable not found")
        print_test("Java Installation", "FAIL")
        return False
    except Exception as e:
        print(f"   Error checking Java: {e}")
        print_test("Java Installation", "FAIL")
        return False

def test_python_packages():
    """Test required Python packages"""
    print_test("Testing Python Packages", "RUNNING")
    
    required_packages = [
        'pyspark',
        'pandas', 
        'numpy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if not missing_packages:
        print_test("Python Packages", "PASS")
        return True
    else:
        print(f"   Missing packages: {missing_packages}")
        print_test("Python Packages", "FAIL")
        return False

def test_pyspark_basic():
    """Test basic PySpark functionality"""
    print_test("Testing PySpark Basic Functionality", "RUNNING")
    
    try:
        from pyspark.sql import SparkSession
        
        # Create SparkSession
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("PySpark_Verification") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        print(f"   ✅ SparkSession created successfully")
        print(f"   ✅ Spark version: {spark.version}")
        print(f"   ✅ Available cores: {spark.sparkContext.defaultParallelism}")
        
        # Test DataFrame creation
        data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        columns = ["name", "age"]
        df = spark.createDataFrame(data, columns)
        
        # Test basic operations
        count = df.count()
        assert count == 3, f"Expected 3 rows, got {count}"
        print(f"   ✅ DataFrame operations working (count: {count})")
        
        # Test SQL functionality
        df.createOrReplaceTempView("people")
        result = spark.sql("SELECT name FROM people WHERE age > 25")
        result_count = result.count()
        assert result_count == 2, f"Expected 2 rows from SQL query, got {result_count}"
        print(f"   ✅ Spark SQL working (filtered count: {result_count})")
        
        spark.stop()
        print_test("PySpark Basic Functionality", "PASS")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        print_test("PySpark Basic Functionality", "FAIL")
        return False

def test_mllib_functionality():
    """Test MLlib machine learning functionality"""
    print_test("Testing MLlib Functionality", "RUNNING")
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
        from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
        from pyspark.ml.regression import LinearRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
        from pyspark.ml import Pipeline
        import pyspark.sql.functions as F
        
        print(f"   ✅ MLlib imports successful")
        
        # Create SparkSession
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("MLlib_Verification") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Create sample dataset for ML testing
        data = [
            (1.0, 2.0, 3.0, 1.0, "A", 1),
            (2.0, 3.0, 4.0, 2.0, "B", 0),
            (3.0, 4.0, 5.0, 1.5, "A", 1),
            (4.0, 5.0, 6.0, 2.5, "B", 0),
            (5.0, 6.0, 7.0, 3.0, "A", 1),
            (6.0, 7.0, 8.0, 3.5, "B", 0),
            (7.0, 8.0, 9.0, 4.0, "A", 1),
            (8.0, 9.0, 10.0, 4.5, "B", 0)
        ]
        columns = ["feature1", "feature2", "feature3", "numeric_feature", "category", "label"]
        df = spark.createDataFrame(data, columns)
        
        print(f"   ✅ Test dataset created ({df.count()} rows)")
        
        # Test VectorAssembler
        assembler = VectorAssembler(
            inputCols=["feature1", "feature2", "feature3", "numeric_feature"],
            outputCol="features"
        )
        df_assembled = assembler.transform(df)
        print(f"   ✅ VectorAssembler working")
        
        # Test StringIndexer
        indexer = StringIndexer(inputCol="category", outputCol="category_index")
        df_indexed = indexer.fit(df_assembled).transform(df_assembled)
        print(f"   ✅ StringIndexer working")
        
        # Test StandardScaler
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(df_indexed)
        df_scaled = scaler_model.transform(df_indexed)
        print(f"   ✅ StandardScaler working")
        
        # Test Logistic Regression
        lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
        lr_model = lr.fit(df_scaled)
        print(f"   ✅ LogisticRegression training successful")
        
        # Test predictions
        predictions = lr_model.transform(df_scaled)
        prediction_count = predictions.count()
        assert prediction_count == df.count(), "Prediction count mismatch"
        print(f"   ✅ Predictions generated ({prediction_count} rows)")
        
        # Test evaluation
        evaluator = BinaryClassificationEvaluator(labelCol="label")
        auc = evaluator.evaluate(predictions)
        print(f"   ✅ Model evaluation working (AUC: {auc:.3f})")
        
        # Test Random Forest
        rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=3)
        rf_model = rf.fit(df_scaled)
        print(f"   ✅ RandomForestClassifier working")
        
        # Test Linear Regression
        linear_reg = LinearRegression(featuresCol="scaled_features", labelCol="feature1")
        linear_model = linear_reg.fit(df_scaled)
        print(f"   ✅ LinearRegression working")
        
        # Test Pipeline
        pipeline = Pipeline(stages=[assembler, indexer, scaler, lr])
        pipeline_model = pipeline.fit(df)
        print(f"   ✅ ML Pipeline working")
        
        spark.stop()
        print_test("MLlib Functionality", "PASS")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        print_test("MLlib Functionality", "FAIL")
        try:
            spark.stop()
        except:
            pass
        return False

def test_air_traffic_simulation():
    """Test with air traffic-like data simulation"""
    print_test("Testing Air Traffic Data Simulation", "RUNNING")
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        import pyspark.sql.functions as F
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
        import random
        
        # Create SparkSession
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("AirTraffic_Simulation") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Generate sample air traffic data
        carriers = ["AA", "DL", "UA", "WN", "B6"]
        origins = ["JFK", "LAX", "ORD", "DFW", "ATL"]
        destinations = ["LAS", "MIA", "SEA", "BOS", "SFO"]
        
        # Create synthetic air traffic data
        air_traffic_data = []
        for i in range(100):
            month = random.randint(1, 12)
            day_of_week = random.randint(1, 7)
            distance = random.randint(200, 3000)
            dep_delay = random.randint(-20, 120)
            arr_delay = random.randint(-30, 150)
            carrier = random.choice(carriers)
            origin = random.choice(origins)
            dest = random.choice(destinations)
            
            # Create binary delay label (1 if arrival delay > 15 minutes)
            delayed = 1 if arr_delay > 15 else 0
            
            air_traffic_data.append((
                month, day_of_week, distance, dep_delay, arr_delay,
                carrier, origin, dest, delayed
            ))
        
        columns = ["Month", "DayOfWeek", "Distance", "DepDelay", "ArrDelay",
                  "UniqueCarrier", "Origin", "Dest", "Delayed"]
        
        df = spark.createDataFrame(air_traffic_data, columns)
        print(f"   ✅ Synthetic air traffic data created ({df.count()} flights)")
        
        # Test feature engineering for delay prediction
        feature_cols = ["Month", "DayOfWeek", "Distance", "DepDelay"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_features = assembler.transform(df)
        
        # Filter out null values
        df_clean = df_features.filter(
            F.col("features").isNotNull() & F.col("Delayed").isNotNull()
        )
        
        print(f"   ✅ Feature engineering completed")
        
        # Split data for training
        train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)
        
        # Train delay prediction model
        lr = LogisticRegression(featuresCol="features", labelCol="Delayed")
        model = lr.fit(train_data)
        
        # Make predictions
        predictions = model.transform(test_data)
        
        # Evaluate model
        evaluator = BinaryClassificationEvaluator(labelCol="Delayed")
        auc = evaluator.evaluate(predictions)
        
        print(f"   ✅ Delay prediction model trained")
        print(f"   ✅ Model AUC score: {auc:.3f}")
        print(f"   ✅ Training data: {train_data.count()} flights")
        print(f"   ✅ Test data: {test_data.count()} flights")
        
        # Test aggregations similar to air traffic analysis
        delay_stats = df.groupBy("UniqueCarrier").agg(
            F.avg("ArrDelay").alias("avg_delay"),
            F.count("*").alias("flight_count")
        ).orderBy("avg_delay", ascending=False)
        
        print(f"   ✅ Carrier delay analysis completed")
        print(f"   ✅ Analysis covers {delay_stats.count()} carriers")
        
        spark.stop()
        print_test("Air Traffic Data Simulation", "PASS")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        print_test("Air Traffic Data Simulation", "FAIL")
        try:
            spark.stop()
        except:
            pass
        return False

def test_performance_benchmark():
    """Run a simple performance benchmark"""
    print_test("Testing Performance Benchmark", "RUNNING")
    
    try:
        from pyspark.sql import SparkSession
        import time
        
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("Performance_Benchmark") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Create a larger dataset for performance testing
        start_time = time.time()
        
        # Generate 10,000 rows of data
        data = [(i, i*2, i*3, f"category_{i%10}") for i in range(10000)]
        columns = ["id", "value1", "value2", "category"]
        df = spark.createDataFrame(data, columns)
        
        # Force computation
        count = df.count()
        
        # Perform some operations
        result = df.groupBy("category").agg(
            {"value1": "avg", "value2": "sum"}
        ).collect()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   ✅ Processed {count} rows in {duration:.2f} seconds")
        print(f"   ✅ Throughput: {count/duration:.0f} rows/second")
        print(f"   ✅ Grouped into {len(result)} categories")
        
        spark.stop()
        print_test("Performance Benchmark", "PASS")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        print_test("Performance Benchmark", "FAIL")
        try:
            spark.stop()
        except:
            pass
        return False

def main():
    """Main verification function"""
    print_header("PySpark & MLlib Installation Verification")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {os.name}")
    
    # Track test results
    tests = []
    
    # Run all tests
    tests.append(("Java Installation", test_java_installation()))
    tests.append(("Python Packages", test_python_packages()))
    tests.append(("PySpark Basic", test_pyspark_basic()))
    tests.append(("MLlib Functionality", test_mllib_functionality()))
    tests.append(("Air Traffic Simulation", test_air_traffic_simulation()))
    tests.append(("Performance Benchmark", test_performance_benchmark()))
    
    # Print summary
    print_header("VERIFICATION SUMMARY")
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        if result:
            print_test(f"{test_name}", "PASS")
            passed += 1
        else:
            print_test(f"{test_name}", "FAIL")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! PySpark and MLlib are correctly installed.")
        print("✅ Your environment is ready for air traffic data analysis!")
        print("\n🚀 Next steps:")
        print("   1. Start Jupyter Lab: jupyter lab")
        print("   2. Open AirTrafficProcessor.ipynb")
        print("   3. Run the first cell to initialize SparkSession")
        print("   4. Follow the DataFrame and MLlib exercises")
        return True
    else:
        print(f"\n❌ {failed} tests failed. Please check the installation.")
        print("\n🔧 Troubleshooting tips:")
        if not tests[0][1]:  # Java test failed
            print("   • Install Java: sudo apt install openjdk-11-jdk")
            print("   • Set JAVA_HOME in ~/.bashrc")
        if not tests[1][1]:  # Python packages test failed
            print("   • Install missing packages: pip install pyspark pandas numpy matplotlib")
        if not tests[2][1]:  # PySpark basic test failed
            print("   • Check Java installation and JAVA_HOME")
            print("   • Verify PySpark installation: pip show pyspark")
        print("   • Refer to README.md for detailed setup instructions")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
