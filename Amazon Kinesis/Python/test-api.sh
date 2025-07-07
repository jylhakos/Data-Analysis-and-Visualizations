#!/bin/bash
# Test script for IoT Temperature Forecasting API

# Configuration
if [ -z "$1" ]; then
    echo "Usage: $0 <API_BASE_URL>"
    echo "Example: $0 http://localhost:8000"
    echo "Example: $0 http://iot-temp-forecast-alb-123456789.us-east-1.elb.amazonaws.com"
    exit 1
fi

API_BASE_URL=$1
echo "🧪 Testing IoT Temperature Forecasting API"
echo "Base URL: $API_BASE_URL"
echo ""

# Test 1: Health Check
echo "1️⃣ Testing health check..."
response=$(curl -s -w "HTTP_STATUS:%{http_code}" "$API_BASE_URL/")
http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
if [ "$http_status" = "200" ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed (HTTP $http_status)"
    exit 1
fi

# Test 2: Submit Sensor Data
echo ""
echo "2️⃣ Testing sensor data submission..."
sensor_data='{
  "sensor_id": "test-sensor-001",
  "temperature": 25.5,
  "humidity": 60.0,
  "location": "TestLab"
}'

response=$(curl -s -w "HTTP_STATUS:%{http_code}" \
  -X POST "$API_BASE_URL/sensor-data" \
  -H "Content-Type: application/json" \
  -d "$sensor_data")

http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
if [ "$http_status" = "200" ]; then
    echo "✅ Sensor data submission passed"
else
    echo "❌ Sensor data submission failed (HTTP $http_status)"
    echo "Response: $(echo $response | sed 's/HTTP_STATUS:[0-9]*//')"
fi

# Test 3: Generate More Test Data
echo ""
echo "3️⃣ Generating more test data..."
for i in {1..20}; do
    temp=$(echo "scale=1; 20 + ($i % 10) + (($RANDOM % 100) - 50) / 10" | bc -l)
    humidity=$(echo "scale=1; 50 + (($RANDOM % 300) - 150) / 10" | bc -l)
    
    test_data='{
      "sensor_id": "test-sensor-001",
      "temperature": '$temp',
      "humidity": '$humidity',
      "location": "TestLab"
    }'
    
    curl -s -X POST "$API_BASE_URL/sensor-data" \
      -H "Content-Type: application/json" \
      -d "$test_data" > /dev/null
    
    echo -n "."
done
echo ""
echo "✅ Generated 20 additional data points"

# Test 4: List Sensors
echo ""
echo "4️⃣ Testing sensor list..."
response=$(curl -s -w "HTTP_STATUS:%{http_code}" "$API_BASE_URL/sensors")
http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
if [ "$http_status" = "200" ]; then
    echo "✅ Sensor list retrieval passed"
    echo "Response: $(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.sensors')"
else
    echo "❌ Sensor list retrieval failed (HTTP $http_status)"
fi

# Test 5: Get Historical Data
echo ""
echo "5️⃣ Testing historical data retrieval..."
response=$(curl -s -w "HTTP_STATUS:%{http_code}" "$API_BASE_URL/historical-data?sensor_id=test-sensor-001&limit=5")
http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
if [ "$http_status" = "200" ]; then
    echo "✅ Historical data retrieval passed"
    data_count=$(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.count')
    echo "Retrieved $data_count data points"
else
    echo "❌ Historical data retrieval failed (HTTP $http_status)"
fi

# Test 6: Check Model Status
echo ""
echo "6️⃣ Testing model status..."
response=$(curl -s -w "HTTP_STATUS:%{http_code}" "$API_BASE_URL/model/status")
http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
if [ "$http_status" = "200" ]; then
    echo "✅ Model status check passed"
    is_trained=$(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.is_trained')
    data_points=$(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.available_data_points')
    echo "Model trained: $is_trained, Data points: $data_points"
else
    echo "❌ Model status check failed (HTTP $http_status)"
fi

# Test 7: Train Model (if enough data)
echo ""
echo "7️⃣ Testing model training..."
if [ "$data_points" -ge 100 ]; then
    response=$(curl -s -w "HTTP_STATUS:%{http_code}" -X POST "$API_BASE_URL/model/train")
    http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    if [ "$http_status" = "200" ]; then
        echo "✅ Model training initiated"
        echo "Response: $(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.message')"
    else
        echo "❌ Model training failed (HTTP $http_status)"
    fi
else
    echo "⏭️ Skipping model training (need at least 100 data points, have $data_points)"
fi

# Test 8: Get Forecast (if model is trained)
echo ""
echo "8️⃣ Testing temperature forecast..."
# Wait a bit for training to potentially complete
sleep 2

# Check if model is trained
response=$(curl -s "$API_BASE_URL/model/status")
is_trained=$(echo $response | jq -r '.is_trained')

if [ "$is_trained" = "true" ]; then
    forecast_request='{
      "sensor_id": "test-sensor-001",
      "hours_ahead": 24
    }'
    
    response=$(curl -s -w "HTTP_STATUS:%{http_code}" \
      -X POST "$API_BASE_URL/forecast" \
      -H "Content-Type: application/json" \
      -d "$forecast_request")
    
    http_status=$(echo $response | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    if [ "$http_status" = "200" ]; then
        echo "✅ Temperature forecast generation passed"
        predictions_count=$(echo $response | sed 's/HTTP_STATUS:[0-9]*//' | jq -r '.predictions | length')
        echo "Generated $predictions_count predictions"
    else
        echo "❌ Temperature forecast generation failed (HTTP $http_status)"
        echo "Response: $(echo $response | sed 's/HTTP_STATUS:[0-9]*//')"
    fi
else
    echo "⏭️ Skipping forecast test (model not trained yet)"
fi

echo ""
echo "🎉 API testing completed!"
echo ""
echo "📊 Access API documentation at: $API_BASE_URL/docs"
echo "🔍 For more detailed testing, visit the interactive docs above"
