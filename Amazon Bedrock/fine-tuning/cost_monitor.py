#!/usr/bin/env python3
"""
Amazon Bedrock Cost monitor
==========================

This script monitors and reports on Amazon Bedrock usage and costs,
helping you stay within budget during BERT fine-tuning operations.

Features:
- Real-time cost monitoring
- Budget tracking
- Usage analytics
- Cost optimization recommendations
- Alert integration

Usage:
    python cost_monitor.py --check-current
    python cost_monitor.py --set-alert 100
    python cost_monitor.py --report monthly
"""

import boto3
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockCostMonitor:
    """Monitor Amazon Bedrock costs and usage"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.ce_client = boto3.client('ce', region_name='us-east-1')  # Cost Explorer is only in us-east-1
        self.cloudwatch = boto3.client('cloudwatch', region_name=region_name)
        self.bedrock = boto3.client('bedrock', region_name=region_name)
        self.budgets = boto3.client('budgets', region_name='us-east-1')  # Budgets is only in us-east-1
        
    def get_current_costs(self, days: int = 30) -> Dict:
        """Get current Bedrock costs for the specified period"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost', 'UsageQuantity'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Bedrock']
                    }
                }
            )
            
            total_cost = 0
            daily_costs = []
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                if result['Groups']:
                    cost = float(result['Groups'][0]['Metrics']['BlendedCost']['Amount'])
                    usage = float(result['Groups'][0]['Metrics']['UsageQuantity']['Amount'])
                else:
                    cost = 0
                    usage = 0
                
                total_cost += cost
                daily_costs.append({
                    'date': date,
                    'cost': cost,
                    'usage': usage
                })
            
            return {
                'total_cost': total_cost,
                'daily_costs': daily_costs,
                'period_days': days,
                'average_daily_cost': total_cost / days if days > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting costs: {e}")
            return {'total_cost': 0, 'daily_costs': [], 'period_days': days, 'average_daily_cost': 0}
    
    def get_bedrock_usage(self) -> Dict:
        """Get Bedrock usage statistics"""
        try:
            # Get model customization jobs
            jobs_response = self.bedrock.list_model_customization_jobs(maxResults=50)
            jobs = jobs_response.get('modelCustomizationJobSummaries', [])
            
            # Analyze job statuses
            job_stats = {
                'total_jobs': len(jobs),
                'completed_jobs': len([j for j in jobs if j['status'] == 'Completed']),
                'failed_jobs': len([j for j in jobs if j['status'] == 'Failed']),
                'running_jobs': len([j for j in jobs if j['status'] == 'InProgress']),
                'recent_jobs': []
            }
            
            # Get recent jobs
            for job in jobs[:5]:  # Last 5 jobs
                job_stats['recent_jobs'].append({
                    'name': job['jobName'],
                    'status': job['status'],
                    'creation_time': job['creationTime'].isoformat(),
                    'base_model': job.get('baseModelArn', 'Unknown')
                })
            
            return job_stats
            
        except Exception as e:
            logger.error(f"Error getting Bedrock usage: {e}")
            return {'total_jobs': 0, 'completed_jobs': 0, 'failed_jobs': 0, 'running_jobs': 0, 'recent_jobs': []}
    
    def check_budget_status(self, budget_name: str = None) -> Dict:
        """Check budget status and alerts"""
        try:
            account_id = boto3.client('sts').get_caller_identity()['Account']
            
            if budget_name is None:
                # List all budgets and find Bedrock-related ones
                budgets_response = self.budgets.describe_budgets(AccountId=account_id)
                bedrock_budgets = [
                    b for b in budgets_response['Budgets'] 
                    if 'bedrock' in b['BudgetName'].lower() or 'bert' in b['BudgetName'].lower()
                ]
                
                if not bedrock_budgets:
                    return {'status': 'No Bedrock budgets found'}
                
                budget = bedrock_budgets[0]  # Use first matching budget
            else:
                budget_response = self.budgets.describe_budget(
                    AccountId=account_id,
                    BudgetName=budget_name
                )
                budget = budget_response['Budget']
            
            budget_info = {
                'budget_name': budget['BudgetName'],
                'budget_limit': budget['BudgetLimit']['Amount'],
                'budget_unit': budget['BudgetLimit']['Unit'],
                'time_unit': budget['TimeUnit'],
                'actual_spend': budget.get('CalculatedSpend', {}).get('ActualSpend', {}).get('Amount', '0'),
                'forecasted_spend': budget.get('CalculatedSpend', {}).get('ForecastedSpend', {}).get('Amount', '0')
            }
            
            # Calculate percentage used
            limit = float(budget_info['budget_limit'])
            actual = float(budget_info['actual_spend'])
            percentage_used = (actual / limit * 100) if limit > 0 else 0
            
            budget_info['percentage_used'] = percentage_used
            budget_info['status'] = 'OK' if percentage_used < 80 else 'WARNING' if percentage_used < 100 else 'EXCEEDED'
            
            return budget_info
            
        except Exception as e:
            logger.error(f"Error checking budget: {e}")
            return {'status': 'Error checking budget', 'error': str(e)}
    
    def get_cost_optimization_recommendations(self, costs: Dict, usage: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check average daily cost
        avg_daily = costs.get('average_daily_cost', 0)
        if avg_daily > 10:
            recommendations.append(
                f"High daily average cost (${avg_daily:.2f}). Consider using smaller models or reducing training frequency."
            )
        
        # Check failed jobs
        failed_jobs = usage.get('failed_jobs', 0)
        total_jobs = usage.get('total_jobs', 1)
        failure_rate = failed_jobs / total_jobs if total_jobs > 0 else 0
        
        if failure_rate > 0.2:
            recommendations.append(
                f"High job failure rate ({failure_rate:.1%}). Review training data quality and hyperparameters."
            )
        
        # Check running jobs
        running_jobs = usage.get('running_jobs', 0)
        if running_jobs > 3:
            recommendations.append(
                f"Multiple jobs running ({running_jobs}). Consider queuing jobs to avoid parallel costs."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "Monitor costs daily during active training periods",
                "Use smaller datasets for initial experiments",
                "Set up billing alerts for early warning",
                "Clean up unused model artifacts in S3"
            ])
        
        return recommendations
    
    def generate_report(self, days: int = 30) -> Dict:
        """Generate comprehensive cost and usage report"""
        logger.info(f"Generating {days}-day cost report...")
        
        costs = self.get_current_costs(days)
        usage = self.get_bedrock_usage()
        budget = self.check_budget_status()
        recommendations = self.get_cost_optimization_recommendations(costs, usage)
        
        report = {
            'report_date': datetime.now().isoformat(),
            'period_days': days,
            'costs': costs,
            'usage': usage,
            'budget': budget,
            'recommendations': recommendations,
            'summary': {
                'total_cost': costs['total_cost'],
                'daily_average': costs['average_daily_cost'],
                'total_jobs': usage['total_jobs'],
                'success_rate': (usage['completed_jobs'] / usage['total_jobs'] * 100) if usage['total_jobs'] > 0 else 0
            }
        }
        
        return report
    
    def create_cost_alert(self, threshold_amount: float, email: str) -> bool:
        """Create a cost alert for Bedrock usage"""
        try:
            account_id = boto3.client('sts').get_caller_identity()['Account']
            
            # Create SNS topic for alerts
            sns = boto3.client('sns', region_name=self.region_name)
            topic_response = sns.create_topic(Name='bedrock-cost-alerts')
            topic_arn = topic_response['TopicArn']
            
            # Subscribe email to topic
            sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            
            # Create CloudWatch alarm
            alarm_name = f'bedrock-cost-alert-{threshold_amount}'
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='EstimatedCharges',
                Namespace='AWS/Billing',
                Period=86400,  # 24 hours
                Statistic='Maximum',
                Threshold=threshold_amount,
                ActionsEnabled=True,
                AlarmActions=[topic_arn],
                AlarmDescription=f'Alert when Bedrock costs exceed ${threshold_amount}',
                Dimensions=[
                    {
                        'Name': 'Currency',
                        'Value': 'USD'
                    },
                    {
                        'Name': 'ServiceName',
                        'Value': 'AmazonBedrock'
                    }
                ]
            )
            
            logger.info(f"Cost alert created: ${threshold_amount} threshold")
            return True
            
        except Exception as e:
            logger.error(f"Error creating cost alert: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Amazon Bedrock Cost Monitor')
    parser.add_argument('--check-current', action='store_true', help='Check current costs')
    parser.add_argument('--report', choices=['daily', 'weekly', 'monthly'], help='Generate cost report')
    parser.add_argument('--set-alert', type=float, metavar='AMOUNT', help='Set cost alert threshold')
    parser.add_argument('--email', help='Email for alerts')
    parser.add_argument('--budget-name', help='Specific budget name to check')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    monitor = BedrockCostMonitor(region_name=args.region)
    
    if args.check_current:
        print("="*60)
        print("CURRENT BEDROCK COSTS")
        print("="*60)
        
        costs = monitor.get_current_costs(days=7)  # Last 7 days
        usage = monitor.get_bedrock_usage()
        budget = monitor.check_budget_status(args.budget_name)
        
        print(f"Last 7 days total cost: ${costs['total_cost']:.2f}")
        print(f"Average daily cost: ${costs['average_daily_cost']:.2f}")
        print(f"Total training jobs: {usage['total_jobs']}")
        print(f"Running jobs: {usage['running_jobs']}")
        
        if 'budget_name' in budget:
            print(f"Budget: {budget['budget_name']} - {budget['percentage_used']:.1f}% used")
            print(f"Status: {budget['status']}")
    
    if args.report:
        days_map = {'daily': 1, 'weekly': 7, 'monthly': 30}
        days = days_map[args.report]
        
        print(f"="*60)
        print(f"BEDROCK COST REPORT ({args.report.upper()})")
        print("="*60)
        
        report = monitor.generate_report(days)
        
        print(f"Report Period: {days} days")
        print(f"Total Cost: ${report['summary']['total_cost']:.2f}")
        print(f"Daily Average: ${report['summary']['daily_average']:.2f}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nFull report saved to: {args.output}")
    
    if args.set_alert:
        if not args.email:
            print("Error: --email is required when setting alerts")
            return
        
        print(f"Setting cost alert for ${args.set_alert}")
        success = monitor.create_cost_alert(args.set_alert, args.email)
        if success:
            print("✅ Cost alert created successfully")
        else:
            print("❌ Failed to create cost alert")

if __name__ == "__main__":
    main()
