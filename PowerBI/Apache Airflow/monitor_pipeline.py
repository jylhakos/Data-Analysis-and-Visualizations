#!/usr/bin/env python3
"""
Fish weight prediction pipeline monitoring script
================================================

This script monitors the Apache Airflow fish weight prediction pipeline and provides status updates.
It can be used to check pipeline health, view recent runs, and generate fish model performance reports.

Author: Fish ML Pipeline Team
Date: July 2025
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

class FishPipelineMonitor:
    """Monitor class for the fish weight prediction pipeline."""
    
    def __init__(self, airflow_home=None):
        self.airflow_home = airflow_home or os.environ.get('AIRFLOW_HOME', './airflow_home')
        self.data_dir = os.path.join(self.airflow_home, 'ml_data')
        self.model_dir = os.path.join(self.airflow_home, 'ml_models')
        
    def check_airflow_status(self):
        """Check if Airflow services are running."""
        print("🔍 Checking Fish Weight Prediction Airflow Status")
        print("=" * 50)
        
        # Check for Airflow processes
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'airflow'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                processes = result.stdout.strip().split('\n')
                print(f"✅ Found {len(processes)} Airflow processes running")
                
                # Check specific services
                webserver_result = subprocess.run(['pgrep', '-f', 'airflow-webserver'], 
                                                capture_output=True, text=True)
                scheduler_result = subprocess.run(['pgrep', '-f', 'airflow-scheduler'], 
                                                capture_output=True, text=True)
                
                if webserver_result.returncode == 0:
                    print("✅ Webserver is running")
                else:
                    print("❌ Webserver is not running")
                    
                if scheduler_result.returncode == 0:
                    print("✅ Scheduler is running")
                else:
                    print("❌ Scheduler is not running")
            else:
                print("❌ No Airflow processes found")
                return False
                
        except FileNotFoundError:
            print("❌ Unable to check processes (pgrep not available)")
            return False
            
        # Check if web UI is accessible
        try:
            import requests
            response = requests.get('http://localhost:8080/health', timeout=5)
            if response.status_code == 200:
                print("✅ Web UI is accessible at http://localhost:8080")
            else:
                print(f"⚠️  Web UI responded with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Web UI is not accessible: {str(e)}")
            
        return True
    
    def check_pipeline_files(self):
        """Check if fish pipeline files and directories exist."""
        print("\n📁 Checking Fish Pipeline Files")
        print("=" * 45)
        
        # Check directories
        dirs_to_check = [self.airflow_home, self.data_dir, self.model_dir]
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                print(f"✅ {dir_path}")
            else:
                print(f"❌ {dir_path} (missing)")
                
        # Check important fish-related files
        files_to_check = [
            os.path.join(self.airflow_home, 'airflow.cfg'),
            os.path.join(self.airflow_home, 'dags', 'ml_pipeline_dag.py'),
            'Dataset/Fish.csv',
            'fish_supervised_pipeline.py',
            'fish_regression.py',
            'fish_analysis.py',
            'requirements.txt'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} (missing)")
    
    def get_recent_runs(self, days=7):
        """Get information about recent fish pipeline runs."""
        print(f"\n📊 Recent Fish Weight Prediction Runs (Last {days} days)")
        print("=" * 60)
        
        # Check for fish model files with timestamps
        if os.path.exists(self.model_dir):
            model_files = []
            for file in os.listdir(self.model_dir):
                if file.endswith('.json') and 'fish' in file.lower():
                    file_path = os.path.join(self.model_dir, file)
                    mtime = os.path.getmtime(file_path)
                    model_files.append((file, datetime.fromtimestamp(mtime)))
            
            if model_files:
                model_files.sort(key=lambda x: x[1], reverse=True)
                print("Recent fish model artifacts:")
                for file, mtime in model_files[:5]:
                    print(f"  📄 {file} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("❌ No fish model artifacts found")
        else:
            print("❌ Model directory not found")
            
        # Check for fish data files
        if os.path.exists(self.data_dir):
            data_files = []
            for file in os.listdir(self.data_dir):
                if 'fish' in file.lower():
                    file_path = os.path.join(self.data_dir, file)
                    mtime = os.path.getmtime(file_path)
                    data_files.append((file, datetime.fromtimestamp(mtime)))
            
            if data_files:
                data_files.sort(key=lambda x: x[1], reverse=True)
                print("\nRecent fish data artifacts:")
                for file, mtime in data_files[:5]:
                    print(f"  📄 {file} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("❌ No fish data artifacts found")
        else:
            print("❌ Data directory not found")
    
    def get_model_performance(self):
        """Get the latest fish weight prediction model performance metrics."""
        print("\n🎯 Latest Fish Weight Prediction Performance")
        print("=" * 50)
        
        eval_file = os.path.join(self.model_dir, 'fish_evaluation_results.json')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    evaluation = json.load(f)
                
                print(f"� Fish Weight Prediction Metrics:")
                print(f"   R²: {evaluation.get('test_r2', 'N/A'):.4f}")
                print(f"   RMSE: {evaluation.get('test_rmse', 'N/A'):.2f}g")
                print(f"   MAE: {evaluation.get('test_mae', 'N/A'):.2f}g")
                print(f"   MSE: {evaluation.get('test_mse', 'N/A'):.2f}")
                print(f"   MAPE: {evaluation.get('test_mape', 'N/A'):.2f}%")
                
                eval_date = evaluation.get('evaluation_date', 'Unknown')
                print(f"   Evaluated: {eval_date}")
                
                # Performance assessment for fish weight prediction
                r2_score = evaluation.get('test_r2', 0)
                if r2_score > 0.9:
                    print("🎯 Outstanding fish weight prediction!")
                elif r2_score > 0.85:
                    print("🎯 Excellent fish weight prediction!")
                elif r2_score > 0.7:
                    print("👍 Good fish weight prediction!")
                elif r2_score > 0.5:
                    print("⚡ Moderate fish weight prediction!")
                else:
                    print("📈 Fish weight prediction needs improvement!")
                    
            except Exception as e:
                print(f"❌ Error reading fish evaluation results: {str(e)}")
        else:
            print("❌ No fish evaluation results found")
            
        # Check fish training results
        training_file = os.path.join(self.model_dir, 'fish_training_results.json')
        if os.path.exists(training_file):
            try:
                with open(training_file, 'r') as f:
                    training = json.load(f)
                
                print(f"\n🏆 Best Fish Model: {training.get('best_model', 'Unknown')}")
                print(f"   CV Score: {training.get('best_cv_score', 'N/A'):.4f}")
                print(f"   Dataset: {training.get('dataset', 'Fish.csv')}")
                
            except Exception as e:
                print(f"❌ Error reading fish training results: {str(e)}")
    
    def generate_status_report(self):
        """Generate a comprehensive fish weight prediction status report."""
        print("� Fish Weight Prediction Pipeline Status Report")
        print("=" * 65)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Airflow Home: {self.airflow_home}")
        
        # Check all components
        self.check_airflow_status()
        self.check_pipeline_files()
        self.get_recent_runs()
        self.get_model_performance()
        
        print("\n" + "=" * 60)
        print("📋 Summary")
        print("=" * 60)
        
        # Quick health check
        issues = []
        
        if not os.path.exists(self.airflow_home):
            issues.append("Airflow home directory missing")
            
        if not os.path.exists(os.path.join(self.model_dir, 'fish_evaluation_results.json')):
            issues.append("No recent fish model evaluation found")
            
        if not os.path.exists('Dataset/Fish.csv'):
            issues.append("Fish.csv dataset missing")
        
        if issues:
            print("⚠️  Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ All fish weight prediction systems operational!")
            
        print("\n💡 Useful commands:")
        print("   - Run fish analysis: python fish_analysis.py")
        print("   - Run fish regression: python fish_regression.py")
        print("   - Run fish pipeline: python fish_supervised_pipeline.py")
        print("   - Start Airflow: ./quickstart.sh")
        print("   - View web UI: http://localhost:8080")
        print("   - Stop Airflow: pkill -f airflow")
        print("   - View logs: tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log")

def main():
    """Main function to run the fish pipeline monitoring script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Fish Weight Prediction Pipeline')
    parser.add_argument('--airflow-home', help='Airflow home directory')
    parser.add_argument('--action', choices=['status', 'performance', 'files'], 
                       default='status', help='Action to perform')
    
    args = parser.parse_args()
    
    monitor = FishPipelineMonitor(args.airflow_home)
    
    if args.action == 'status':
        monitor.generate_status_report()
    elif args.action == 'performance':
        monitor.get_model_performance()
    elif args.action == 'files':
        monitor.check_pipeline_files()

if __name__ == "__main__":
    main()
