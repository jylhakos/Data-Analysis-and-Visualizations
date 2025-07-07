#!/bin/bash
# MQTT Temperature Sensor Simulator
# This script simulates IoT temperature sensors sending data to MQTT broker

MQTT_BROKER="localhost"
MQTT_PORT="1883"

echo "🌡️ Starting MQTT Temperature Sensor Simulator"
echo "MQTT Broker: $MQTT_BROKER:$MQTT_PORT"

# Function to simulate temperature sensor
simulate_sensor() {
    local sensor_id=$1
    local location=$2
    local base_temp=$3
    
    while true; do
        # Generate realistic temperature with daily cycle and noise
        hour=$(date +%H)
        # Daily temperature cycle (warmer during day)
        daily_variation=$(echo "scale=2; 5 * sin(($hour - 6) * 3.14159 / 12)" | bc -l)
        # Random noise
        noise=$(echo "scale=2; (($RANDOM % 200) - 100) / 100" | bc -l)
        # Final temperature
        temp=$(echo "scale=2; $base_temp + $daily_variation + $noise" | bc -l)
        
        # Generate humidity (inversely related to temperature)
        humidity=$(echo "scale=2; 80 - ($temp - 15) * 1.5 + (($RANDOM % 100) - 50) / 10" | bc -l)
        
        # Create JSON payload
        timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
        json_payload=$(cat <<EOF
{
  "sensor_id": "$sensor_id",
  "temperature": $temp,
  "humidity": $humidity,
  "timestamp": "$timestamp",
  "location": "$location"
}
EOF
)
        
        # Publish to MQTT
        mosquitto_pub -h $MQTT_BROKER -p $MQTT_PORT -t "sensors/$sensor_id/temperature" -m "$json_payload"
        
        echo "📡 Sensor $sensor_id ($location): ${temp}°C, ${humidity}% humidity"
        
        # Wait 5-15 seconds before next reading
        sleep_time=$((5 + $RANDOM % 10))
        sleep $sleep_time
    done
}

# Check if mosquitto_pub is available
if ! command -v mosquitto_pub &> /dev/null; then
    echo "❌ mosquitto_pub not found. Please install mosquitto-clients:"
    echo "Ubuntu/Debian: sudo apt-get install mosquitto-clients"
    echo "CentOS/RHEL: sudo yum install mosquitto"
    echo "macOS: brew install mosquitto"
    exit 1
fi

# Start multiple sensor simulations in background
echo "🚀 Starting sensor simulations..."

simulate_sensor "sensor_001" "Office_Building_A" 22 &
simulate_sensor "sensor_002" "Warehouse_B" 18 &
simulate_sensor "sensor_003" "Data_Center_C" 25 &
simulate_sensor "sensor_004" "Factory_Floor_D" 28 &
simulate_sensor "sensor_005" "Outdoor_Station_E" 15 &

echo "✅ Started 5 temperature sensors"
echo "📊 Sensors are publishing to topics: sensors/*/temperature"
echo "🛑 Press Ctrl+C to stop all sensors"

# Wait for user to stop
wait
