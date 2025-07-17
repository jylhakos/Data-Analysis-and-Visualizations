#!/usr/bin/env python3
"""
Quick PySpark and MLlib Verification
====================================

A lightweight verification script to quickly test PySpark and MLlib installation.
This is a simplified version of the comprehensive verify_pyspark_mllib.py script.

Usage: python quick_verify.py
"""

def quick_test():
    """Run quick verification tests"""
    print("🚀 Quick PySpark & MLlib Verification")
    print("=" * 40)
    
    try:
        # Test 1: PySpark imports
        print("🔄 Testing PySpark imports...")
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        print("✅ PySpark imports successful")
        
        # Test 2: MLlib imports  
        print("🔄 Testing MLlib imports...")
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        print("✅ MLlib imports successful")
        
        # Test 3: SparkSession creation
        print("🔄 Testing SparkSession creation...")
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("QuickVerification") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()
        print(f"✅ SparkSession created (Spark v{spark.version})")
        print(f"✅ Available cores: {spark.sparkContext.defaultParallelism}")
        
        # Test 4: DataFrame operations
        print("🔄 Testing DataFrame operations...")
        data = [("Flight1", 100, 1), ("Flight2", 200, 0), ("Flight3", 150, 1)]
        df = spark.createDataFrame(data, ["flight", "distance", "delayed"])
        count = df.count()
        print(f"✅ DataFrame created with {count} rows")
        
        # Test 5: MLlib functionality
        print("🔄 Testing MLlib functionality...")
        assembler = VectorAssembler(inputCols=["distance"], outputCol="features")
        df_features = assembler.transform(df)
        
        lr = LogisticRegression(featuresCol="features", labelCol="delayed")
        model = lr.fit(df_features)
        predictions = model.transform(df_features)
        
        evaluator = BinaryClassificationEvaluator(labelCol="delayed")
        auc = evaluator.evaluate(predictions)
        print(f"✅ MLlib model trained (AUC: {auc:.3f})")
        
        spark.stop()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ PySpark and MLlib are working correctly")
        print("🚀 Ready for air traffic data analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Try running the full verification script:")
        print("   python verify_pyspark_mllib.py")
        try:
            spark.stop()
        except:
            pass
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
