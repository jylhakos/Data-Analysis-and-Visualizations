#!/usr/bin/env python3
"""
Deploy BERT Fine-tuning Job to Databricks
==========================================

This script automates the deployment and execution of BERT fine-tuning
jobs on Databricks using the Jobs API.

Features:
- Creates Databricks job definition
- Uploads code and configuration files
- Submits and monitors training job
- Downloads results and logs

Usage:
    python deploy_training_job.py --config config.yaml --cluster-id <cluster-id>
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional
import zipfile
import base64

try:
    from databricks_sdk import WorkspaceClient
    from databricks_sdk.service.jobs import CreateJob, JobSettings, NotebookTask, PythonWheelTask
    from databricks_sdk.service.compute import ClusterSpec
    DATABRICKS_SDK_AVAILABLE = True
except ImportError:
    DATABRICKS_SDK_AVAILABLE = False
    print("Databricks SDK not available. Please install: pip install databricks-sdk")

import yaml
import requests


class DatabricksJobDeployer:
    """Handles deployment of BERT fine-tuning jobs to Databricks"""
    
    def __init__(self, workspace_url: str, token: str):
        if not DATABRICKS_SDK_AVAILABLE:
            raise ImportError("Databricks SDK is required. Install with: pip install databricks-sdk")
        
        self.client = WorkspaceClient(host=workspace_url, token=token)
        self.workspace_url = workspace_url
        self.token = token
    
    def create_job(self, job_config: Dict) -> str:
        """Create a new Databricks job"""
        print("Creating Databricks job...")
        
        job_settings = JobSettings(
            name=job_config['name'],
            max_concurrent_runs=1,
            tasks=[
                {
                    'task_key': 'bert_training',
                    'description': 'BERT fine-tuning task',
                    'python_wheel_task': PythonWheelTask(
                        package_name=job_config['package_name'],
                        entry_point=job_config['entry_point'],
                        parameters=job_config.get('parameters', [])
                    ),
                    'existing_cluster_id': job_config['cluster_id'],
                    'libraries': job_config.get('libraries', []),
                    'timeout_seconds': job_config.get('timeout', 7200)  # 2 hours default
                }
            ],
            email_notifications={
                'on_success': job_config.get('success_emails', []),
                'on_failure': job_config.get('failure_emails', [])
            }
        )
        
        job = self.client.jobs.create(job_settings)
        print(f"Job created with ID: {job.job_id}")
        return str(job.job_id)
    
    def upload_training_script(self, script_path: str, target_path: str) -> None:
        """Upload training script to Databricks workspace"""
        print(f"📤 Uploading {script_path} to {target_path}")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Use workspace API to upload file
        self.client.workspace.upload(
            path=target_path,
            content=content.encode(),
            format='SOURCE',
            overwrite=True
        )
        
        print(f"✅ Script uploaded to {target_path}")
    
    def upload_config_files(self, config_dir: str, target_dir: str) -> None:
        """Upload configuration files to Databricks"""
        print(f"📁 Uploading config files from {config_dir}")
        
        config_path = Path(config_dir)
        for config_file in config_path.glob('*.yaml'):
            target_path = f"{target_dir}/{config_file.name}"
            
            with open(config_file, 'r') as f:
                content = f.read()
            
            self.client.workspace.upload(
                path=target_path,
                content=content.encode(),
                format='SOURCE',
                overwrite=True
            )
            
            print(f"{config_file.name} uploaded")
    
    def run_job(self, job_id: str, parameters: Optional[Dict] = None) -> str:
        """Submit a job run"""
        print(f"▶️  Starting job {job_id}...")
        
        run_params = {}
        if parameters:
            run_params['python_params'] = parameters
        
        run = self.client.jobs.run_now(job_id=int(job_id), **run_params)
        run_id = run.run_id
        
        print(f"Job run started with ID: {run_id}")
        return str(run_id)
    
    def monitor_job(self, run_id: str) -> Dict:
        """Monitor job execution and return final status"""
        print(f"Monitoring job run {run_id}...")
        
        while True:
            run_info = self.client.jobs.get_run(run_id=int(run_id))
            state = run_info.state.life_cycle_state
            
            print(f"  Status: {state}")
            
            if state in ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']:
                result_state = run_info.state.result_state
                print(f"  Final result: {result_state}")
                
                return {
                    'run_id': run_id,
                    'state': state,
                    'result': result_state,
                    'start_time': run_info.start_time,
                    'end_time': run_info.end_time
                }
            
            time.sleep(30)  # Check every 30 seconds
    
    def download_logs(self, run_id: str, output_dir: str) -> None:
        """Download job logs and outputs"""
        print(f"📥 Downloading logs for run {run_id}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Get run output
            run_output = self.client.jobs.get_run_output(run_id=int(run_id))
            
            # Save logs
            with open(output_path / f"run_{run_id}_logs.txt", 'w') as f:
                if run_output.logs:
                    f.write(run_output.logs)
                else:
                    f.write("No logs available")
            
            print(f"Logs saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️  Could not download logs: {e}")


def create_notebook_job(deployer: DatabricksJobDeployer, config: Dict) -> str:
    """Create a job that runs a Databricks notebook"""
    print("Creating notebook-based job...")
    
    # Upload notebook first
    notebook_path = config['training']['notebook_path']
    target_notebook = f"/Users/{config['user_email']}/bert_fine_tuning_notebook"
    
    deployer.upload_training_script(notebook_path, target_notebook)
    
    # Create job configuration
    job_config = {
        'name': f"BERT-FineTuning-{int(time.time())}",
        'notebook_path': target_notebook,
        'cluster_id': config['cluster_id'],
        'parameters': config.get('notebook_parameters', {}),
        'libraries': [
            {'pypi': {'package': 'transformers>=4.30.0'}},
            {'pypi': {'package': 'torch>=2.0.0'}},
            {'pypi': {'package': 'accelerate>=0.20.0'}}
        ]
    }
    
    # Create job with notebook task
    job_settings = {
        'name': job_config['name'],
        'max_concurrent_runs': 1,
        'tasks': [
            {
                'task_key': 'bert_training_notebook',
                'description': 'BERT fine-tuning via notebook',
                'notebook_task': {
                    'notebook_path': target_notebook,
                    'base_parameters': job_config['parameters']
                },
                'existing_cluster_id': job_config['cluster_id'],
                'libraries': job_config['libraries'],
                'timeout_seconds': 7200
            }
        ]
    }
    
    job = deployer.client.jobs.create(job_settings)
    return str(job.job_id)


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy BERT training job to Databricks')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--cluster-id', required=True, help='Databricks cluster ID')
    parser.add_argument('--notebook', action='store_true', help='Deploy as notebook job')
    parser.add_argument('--monitor', action='store_true', help='Monitor job execution')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get Databricks credentials from environment
    workspace_url = os.getenv('DATABRICKS_HOST')
    token = os.getenv('DATABRICKS_TOKEN')
    
    if not workspace_url or not token:
        print("❌ Please set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables")
        return
    
    # Initialize deployer
    deployer = DatabricksJobDeployer(workspace_url, token)
    
    try:
        # Add cluster ID to config
        config['cluster_id'] = args.cluster_id
        
        if args.notebook:
            # Deploy as notebook job
            job_id = create_notebook_job(deployer, config)
        else:
            # Upload training scripts and configs
            script_path = 'databricks_bert_fine_tuning.py'
            target_script = f"/Users/{config.get('user_email', 'user@company.com')}/bert_training.py"
            
            deployer.upload_training_script(script_path, target_script)
            
            # Upload config files
            config_dir = './config'
            target_config_dir = f"/Users/{config.get('user_email', 'user@company.com')}/config"
            deployer.upload_config_files(config_dir, target_config_dir)
            
            # Create job
            job_config = {
                'name': f"BERT-FineTuning-{int(time.time())}",
                'package_name': 'bert_training',
                'entry_point': 'main',
                'cluster_id': args.cluster_id,
                'parameters': [
                    '--config', f"{target_config_dir}/training-config.yaml"
                ],
                'libraries': [
                    {'pypi': {'package': 'transformers>=4.30.0'}},
                    {'pypi': {'package': 'torch>=2.0.0'}},
                    {'pypi': {'package': 'accelerate>=0.20.0'}},
                    {'pypi': {'package': 'pyyaml>=6.0'}}
                ]
            }
            
            job_id = deployer.create_job(job_config)
        
        # Run the job
        run_id = deployer.run_job(job_id)
        
        if args.monitor:
            # Monitor execution
            result = deployer.monitor_job(run_id)
            
            # Download logs
            deployer.download_logs(run_id, './job_outputs')
            
            print(f"\nJob completed with status: {result['result']}")
            print(f"View results in Databricks workspace")
            print(f"Job URL: {workspace_url}/#job/{job_id}/run/{run_id}")
        else:
            print(f"\nJob submitted successfully!")
            print(f"Monitor progress in Databricks workspace")
            print(f"Job URL: {workspace_url}/#job/{job_id}")
    
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
