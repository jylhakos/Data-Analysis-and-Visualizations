#!/usr/bin/env python3
"""
Weather Station Data Simulator
Simulates weather stations sending data via MQTT for testing the ETL pipeline
"""

import json
import time
import random
import asyncio
from datetime import datetime, timezone
from typing import Dict, List
import paho.mqtt.client as mqtt
from dataclasses import dataclass, asdict
import argparse


@dataclass
class WeatherStation:
    """Weather station configuration"""
    station_id: str
    name: str
    latitude: float
    longitude: float
    altitude: float = 0.0


class WeatherDataGenerator:
    """Generate realistic weather data"""
    
    def __init__(self, station: WeatherStation):
        self.station = station
        # Base values for more realistic variations
        self.base_temp = 20.0 + random.uniform(-10, 10)  # Regional variation
        self.base_pressure = 1013.25 + random.uniform(-20, 20)
        
    def generate_reading(self) -> Dict:
        """Generate a single weather reading"""
        # Simulate diurnal temperature variation
        hour = datetime.now().hour
        temp_variation = 5 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2 PM
        
        # Add random variations
        temperature = self.base_temp + temp_variation + random.uniform(-3, 3)
        
        # Humidity inversely related to temperature (roughly)
        base_humidity = 70 - (temperature - 20) * 2
        humidity = max(10, min(100, base_humidity + random.uniform(-15, 15)))
        
        # Pressure varies slowly
        self.base_pressure += random.uniform(-2, 2)
        self.base_pressure = max(980, min(1050, self.base_pressure))
        
        # Wind data
        wind_speed = max(0, random.exponential(10))  # Exponential distribution
        wind_direction = random.uniform(0, 360)
        
        return {
            "station_id": self.station.station_id,
            "station_name": self.station.name,
            "latitude": self.station.latitude,
            "longitude": self.station.longitude,
            "altitude": self.station.altitude,
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 1),
            "pressure": round(self.base_pressure, 1),
            "wind_speed": round(wind_speed, 1),
            "wind_direction": round(wind_direction, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_quality": "good",
            "sensor_status": "online"
        }


class MQTTWeatherSimulator:
    """MQTT client for sending weather station data"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 username: str = None, password: str = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client = mqtt.Client()
        self.connected = False
        
        # Setup MQTT client
        if username and password:
            self.client.username_pw_set(username, password)
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection"""
        if rc == 0:
            print(f"✅ Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.connected = True
        else:
            print(f"❌ Failed to connect to MQTT broker: {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        print(f"⚠️ Disconnected from MQTT broker: {rc}")
        self.connected = False
    
    def _on_publish(self, client, userdata, mid):
        """Handle message publication"""
        print(f"📤 Message {mid} published successfully")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.5)
                timeout -= 0.5
            
            return self.connected
        except Exception as e:
            print(f"❌ Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
    
    def publish_weather_data(self, station_id: str, data: Dict):
        """Publish weather data to MQTT topic"""
        if not self.connected:
            print("❌ Not connected to MQTT broker")
            return False
        
        topic = f"weather/stations/{station_id}/data"
        payload = json.dumps(data, indent=None)
        
        try:
            result = self.client.publish(topic, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📡 Published data for station {station_id}")
                return True
            else:
                print(f"❌ Failed to publish data: {result.rc}")
                return False
        except Exception as e:
            print(f"❌ Error publishing data: {e}")
            return False


def create_weather_stations() -> List[WeatherStation]:
    """Create 5 weather stations within 100km radius"""
    # Central point (example: around a major city)
    center_lat, center_lon = 40.7589, -73.9851  # New York area
    
    stations = []
    
    # Generate stations in a roughly 100km radius
    for i in range(1, 6):
        # Random displacement within ~100km (roughly 1 degree = 111km)
        lat_offset = random.uniform(-0.9, 0.9)  # ~100km
        lon_offset = random.uniform(-0.9, 0.9)  # ~100km
        
        station = WeatherStation(
            station_id=f"WS{i:03d}",
            name=f"Weather Station {i}",
            latitude=round(center_lat + lat_offset, 6),
            longitude=round(center_lon + lon_offset, 6),
            altitude=round(random.uniform(0, 500), 1)  # 0-500m elevation
        )
        stations.append(station)
    
    return stations


async def simulate_weather_stations(duration_minutes: int = 60, interval_seconds: int = 300):
    """
    Simulate weather stations sending data
    
    Args:
        duration_minutes: How long to run the simulation
        interval_seconds: Interval between readings (300s = 5 minutes for testing)
    """
    print("🌦️ Starting Weather Station Simulator")
    print(f"⏱️ Duration: {duration_minutes} minutes")
    print(f"📊 Data interval: {interval_seconds} seconds")
    print()
    
    # Create weather stations
    stations = create_weather_stations()
    print(f"🏭 Created {len(stations)} weather stations:")
    for station in stations:
        print(f"  • {station.station_id}: {station.name} at ({station.latitude:.4f}, {station.longitude:.4f})")
    print()
    
    # Create data generators
    generators = {station.station_id: WeatherDataGenerator(station) for station in stations}
    
    # Setup MQTT client
    mqtt_client = MQTTWeatherSimulator(
        broker_host="localhost",
        broker_port=1883,
        username="weather_user",
        password="weather_pass"
    )
    
    # Connect to MQTT broker
    if not mqtt_client.connect():
        print("❌ Failed to connect to MQTT broker. Make sure it's running!")
        return
    
    print("🚀 Starting data transmission...")
    print()
    
    try:
        end_time = time.time() + (duration_minutes * 60)
        transmission_count = 0
        
        while time.time() < end_time:
            print(f"📡 Transmission #{transmission_count + 1} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Send data from all stations
            for station_id, generator in generators.items():
                data = generator.generate_reading()
                success = mqtt_client.publish_weather_data(station_id, data)
                
                if success:
                    temp = data['temperature']
                    humidity = data['humidity']
                    wind = data['wind_speed']
                    print(f"  ✅ {station_id}: {temp}°C, {humidity}% humidity, {wind} km/h wind")
                else:
                    print(f"  ❌ {station_id}: Failed to send data")
            
            transmission_count += 1
            print(f"  📈 Total transmissions: {transmission_count * len(stations)}")
            print()
            
            # Wait for next interval
            if time.time() < end_time:
                print(f"⏳ Waiting {interval_seconds} seconds until next transmission...")
                await asyncio.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("⏹️ Simulation stopped by user")
    
    finally:
        mqtt_client.disconnect()
        print("🏁 Weather station simulation completed")


def test_mqtt_connection():
    """Test MQTT broker connectivity"""
    print("🔍 Testing MQTT broker connection...")
    
    client = MQTTWeatherSimulator()
    if client.connect():
        print("✅ MQTT broker connection successful")
        
        # Send test message
        test_data = {
            "test": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "MQTT connection test"
        }
        
        if client.publish_weather_data("TEST", test_data):
            print("✅ Test message sent successfully")
        else:
            print("❌ Failed to send test message")
        
        client.disconnect()
        return True
    else:
        print("❌ MQTT broker connection failed")
        print("Make sure the MQTT broker is running:")
        print("  docker-compose up -d mosquitto")
        return False


def main():
    """Main function"""
    import math  # Import here to avoid issues with dataclass
    
    parser = argparse.ArgumentParser(description="Weather Station Data Simulator")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Simulation duration in minutes (default: 60)")
    parser.add_argument("--interval", type=int, default=300,
                       help="Data transmission interval in seconds (default: 300)")
    parser.add_argument("--test", action="store_true",
                       help="Test MQTT connection only")
    
    args = parser.parse_args()
    
    if args.test:
        test_mqtt_connection()
    else:
        try:
            asyncio.run(simulate_weather_stations(args.duration, args.interval))
        except Exception as e:
            print(f"❌ Simulation error: {e}")


if __name__ == "__main__":
    main()
