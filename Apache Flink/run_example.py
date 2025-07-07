#!/usr/bin/env python3
"""
Script to run the complete HTTP monitoring example
"""

import subprocess
import time
import requests
import threading
import random

def generate_test_traffic():
    """Generate test HTTP traffic to FastAPI server"""
    endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/api/users/1",
        "http://localhost:8000/api/users/2",
        "http://localhost:8000/api/data"
    ]
    
    methods_data = [
        ("GET", "http://localhost:8000/"),
        ("GET", "http://localhost:8000/api/users/1"),
        ("POST", "http://localhost:8000/api/data")
    ]
    
    while True:
        try:
            method, url = random.choice(methods_data)
            
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json={"test": "data"})
            
            print(f"Generated {method} request to {url} - Status: {response.status_code}")
            
            # Random delay between requests
            time.sleep(random.uniform(0.1, 2.0))
            
        except Exception as e:
            print(f"Error generating traffic: {e}")
            time.sleep(1)

def main():
    print("Starting HTTP Monitoring Example...")
    
    # Start Docker Compose services
    print("Starting Docker services...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(30)
    
    # Start traffic generator in background
    print("Starting traffic generator...")
    traffic_thread = threading.Thread(target=generate_test_traffic, daemon=True)
    traffic_thread.start()
    
    # Run PyFlink job
    print("Starting PyFlink monitoring job...")
    try:
        subprocess.run(["python", "pyflink_event_processor.py"], check=True)
    except KeyboardInterrupt:
        print("Stopping monitoring...")
    finally:
        # Cleanup
        print("Stopping Docker services...")
        subprocess.run(["docker-compose", "down"], check=True)

if __name__ == "__main__":
    main()