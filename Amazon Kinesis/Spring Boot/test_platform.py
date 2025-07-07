#!/usr/bin/env python3
"""
Test script for the IoT Temperature Analytics Platform
"""

import requests
import json
import time
from datetime import datetime

# Service URLs
SPRING_BOOT_URL = "http://localhost:8080"
ML_SERVICE_URL = "http://localhost:8001"

def test_spring_boot_health():
    """Test Spring Boot application health"""
    try:
        response = requests.get(f"{SPRING_BOOT_URL}/api/temperature/health")
        if response.status_code == 200:
            print("✅ Spring Boot service is healthy")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Spring Boot health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Spring Boot service not reachable: {e}")
        return False

def test_ml_service_health():
    """Test ML service health"""
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health")
        if response.status_code == 200:
            print("✅ ML service is healthy")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ ML service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ML service not reachable: {e}")
        return False

def test_temperature_data_submission():
    """Test submitting temperature data to Spring Boot"""
    try:
        test_data = {
            "sensorId": "TEST_001",
            "temperature": 22.5,
            "humidity": 45.0,
            "location": "Test Lab",
            "timestamp": datetime.now().isoformat(),
            "deviceType": "test_sensor",
            "batteryLevel": 95,
            "signalStrength": -60
        }
        
        response = requests.post(
            f"{SPRING_BOOT_URL}/api/temperature/data",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Temperature data submission successful")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Temperature data submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Temperature data submission error: {e}")
        return False

def test_ml_temperature_analysis():
    """Test ML service temperature analysis"""
    try:
        test_data = {
            "sensor_id": "TEST_001",
            "temperature": 23.0,
            "humidity": 50.0,
            "location": "Test Lab",
            "timestamp": datetime.now().isoformat(),
            "device_type": "test_sensor",
            "battery_level": 94,
            "signal_strength": -62
        }
        
        response = requests.post(
            f"{ML_SERVICE_URL}/api/temperature/analyze",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ ML temperature analysis successful")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ ML temperature analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ML temperature analysis error: {e}")
        return False

def test_sensor_listing():
    """Test listing sensors from Spring Boot"""
    try:
        response = requests.get(f"{SPRING_BOOT_URL}/api/temperature/sensors")
        if response.status_code == 200:
            data = response.json()
            print("✅ Sensor listing successful")
            print(f"   Active sensors: {data.get('count', 0)}")
            print(f"   Sensors: {list(data.get('sensors', []))}")
            return True
        else:
            print(f"❌ Sensor listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Sensor listing error: {e}")
        return False

def test_end_to_end_workflow():
    """Test complete workflow with multiple data points"""
    print("\n🔄 Testing end-to-end workflow...")
    
    # Submit multiple temperature readings
    for i in range(5):
        temp = 20 + i + (i * 0.5)  # Varying temperatures
        test_data = {
            "sensorId": "E2E_TEST_001",
            "temperature": temp,
            "humidity": 40 + i * 2,
            "location": "E2E Test Lab",
            "timestamp": datetime.now().isoformat(),
            "deviceType": "test_sensor",
            "batteryLevel": 100 - i,
            "signalStrength": -60 - i
        }
        
        # Submit to Spring Boot
        response = requests.post(
            f"{SPRING_BOOT_URL}/api/temperature/data",
            json=test_data
        )
        
        if response.status_code == 200:
            print(f"   ✅ Data point {i+1} submitted: {temp}°C")
        else:
            print(f"   ❌ Data point {i+1} failed")
        
        # Also submit to ML service
        ml_data = {
            "sensor_id": "E2E_TEST_001",
            "temperature": temp,
            "humidity": 40 + i * 2,
            "location": "E2E Test Lab",
            "timestamp": datetime.now().isoformat(),
            "device_type": "test_sensor",
            "battery_level": 100 - i,
            "signal_strength": -60 - i
        }
        
        requests.post(f"{ML_SERVICE_URL}/api/temperature/analyze", json=ml_data)
        time.sleep(0.5)  # Small delay between submissions
    
    # Get sensor statistics
    try:
        response = requests.get(f"{SPRING_BOOT_URL}/api/temperature/stats/E2E_TEST_001")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Statistics retrieved:")
            print(f"      Data points: {stats.get('data_points')}")
            print(f"      Average temp: {stats.get('average_temperature')}°C")
            print(f"      Min temp: {stats.get('min_temperature')}°C")
            print(f"      Max temp: {stats.get('max_temperature')}°C")
        else:
            print(f"   ❌ Failed to get statistics")
    except Exception as e:
        print(f"   ❌ Statistics error: {e}")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting IoT Temperature Analytics Platform Tests")
    print("=" * 60)
    
    tests = [
        ("Spring Boot Health Check", test_spring_boot_health),
        ("ML Service Health Check", test_ml_service_health),
        ("Temperature Data Submission", test_temperature_data_submission),
        ("ML Temperature Analysis", test_ml_temperature_analysis),
        ("Sensor Listing", test_sensor_listing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Run end-to-end test
    test_end_to_end_workflow()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The platform is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the services and try again.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
