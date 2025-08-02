# AWS Configuration Template for Bedrock Fine-tuning
{
  "aws_config": {
    "region": "${region}",
    "s3_bucket": "${bucket_name}",
    "bedrock_execution_role_arn": "${role_arn}",
    "endpoints": {
      "bedrock": "https://bedrock.${region}.amazonaws.com",
      "bedrock_runtime": "https://bedrock-runtime.${region}.amazonaws.com",
      "s3": "https://s3.${region}.amazonaws.com"
    },
    "services": {
      "bedrock": {
        "service_name": "bedrock",
        "region": "${region}"
      },
      "s3": {
        "service_name": "s3",
        "region": "${region}",
        "bucket": "${bucket_name}"
      }
    }
  },
  "training_config": {
    "default_hyperparameters": {
      "epoch_count": "3",
      "batch_size": "8",
      "learning_rate": "0.00002",
      "learning_rate_warmup_steps": "0"
    },
    "supported_models": [
      "amazon.titan-text-express-v1",
      "anthropic.claude-v2",
      "anthropic.claude-instant-v1"
    ]
  },
  "monitoring": {
    "cloudwatch_log_group": "/aws/bedrock/bert-fine-tuning",
    "metrics_enabled": true,
    "cost_alerts_enabled": true
  }
}
