# Essential IAM roles for AWS ETL platform

This document outlines the essential IAM roles required for the Weather ETL platform to process real-time data and deploy a React dashboard on AWS.

## Overview

The ETL application requires several IAM roles with specific permissions to interact with various AWS services securely. Each role follows the principle of least privilege, granting only the minimum permissions necessary for the component to function.

## 1. ETL processing role (AWS Glue)

### Purpose
Execute AWS Glue ETL jobs for batch processing weather data.

### Role name
`weather-etl-glue-processing-role`

### Trust policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "glue.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
- **AWS Managed Policies:**
  - `AWSGlueServiceRole`
  
- **Custom Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-data-lake/*",
        "arn:aws:s3:::weather-etl-data-lake"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:GetTable",
        "glue:GetDatabase",
        "glue:GetPartitions",
        "glue:BatchCreatePartition",
        "glue:BatchUpdatePartition",
        "glue:UpdateTable"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

## 2. Stream Processing role (Apache Flink/Kinesis Analytics)

### Purpose
Execute real-time stream processing jobs using Apache Flink.

### Role name
`weather-etl-stream-processing-role`

### Trust policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "kinesisanalytics.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "kafka:DescribeCluster",
        "kafka:GetBootstrapBrokers",
        "kafka:ListClusters"
      ],
      "Resource": "arn:aws:kafka:*:*:cluster/weather-etl-kafka/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kafka-cluster:Connect",
        "kafka-cluster:AlterCluster",
        "kafka-cluster:DescribeCluster"
      ],
      "Resource": "arn:aws:kafka:*:*:cluster/weather-etl-kafka/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kafka-cluster:*Topic*",
        "kafka-cluster:WriteData",
        "kafka-cluster:ReadData"
      ],
      "Resource": "arn:aws:kafka:*:*:topic/weather-etl-kafka/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-data-lake/*",
        "arn:aws:s3:::weather-etl-data-lake"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

## 3. Query service role (Amazon Athena)

### Purpose
Execute queries against the data lake using Amazon Athena.

### Role name
`weather-etl-query-service-role`

### Trust policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "athena:StartQueryExecution",
        "athena:GetQueryExecution",
        "athena:GetQueryResults",
        "athena:GetWorkGroup",
        "athena:ListQueryExecutions"
      ],
      "Resource": [
        "arn:aws:athena:*:*:workgroup/weather-etl-workgroup"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-data-lake/*",
        "arn:aws:s3:::weather-etl-data-lake",
        "arn:aws:s3:::weather-etl-athena-results/*",
        "arn:aws:s3:::weather-etl-athena-results"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:GetTable",
        "glue:GetDatabase",
        "glue:GetPartitions",
        "glue:GetPartition"
      ],
      "Resource": "*"
    }
  ]
}
```

## 4. Application runtime role (ECS Tasks)

### Purpose
Runtime execution role for microservices running on ECS/Fargate.

### Role name
`weather-etl-application-runtime-role`

### Trust policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-data-lake/*",
        "arn:aws:s3:::weather-etl-data-lake"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kafka:DescribeCluster",
        "kafka:GetBootstrapBrokers"
      ],
      "Resource": "arn:aws:kafka:*:*:cluster/weather-etl-kafka/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kafka-cluster:Connect",
        "kafka-cluster:AlterGroup",
        "kafka-cluster:DescribeGroup"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kafka-cluster:ReadData",
        "kafka-cluster:WriteData"
      ],
      "Resource": "arn:aws:kafka:*:*:topic/weather-etl-kafka/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath"
      ],
      "Resource": "arn:aws:ssm:*:*:parameter/weather-etl/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:weather-etl/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords"
      ],
      "Resource": "*"
    }
  ]
}
```

## 5. ECS execution role

### Purpose
Allow ECS to pull container images and manage task execution.

### Role name
`weather-etl-ecs-execution-role`

### Trust policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
- **AWS Managed Policy:**
  - `AmazonECSTaskExecutionRolePolicy`

## 6. Dashboard deployment role (S3/CloudFront)

### Purpose
Deploy React dashboard to S3 with CloudFront distribution.

### Role Name
`weather-etl-dashboard-deployment-role`

### Trust Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "s3.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:PutBucketWebsite",
        "s3:PutBucketPolicy"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-dashboard/*",
        "arn:aws:s3:::weather-etl-dashboard"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudfront:CreateInvalidation",
        "cloudfront:GetDistribution",
        "cloudfront:UpdateDistribution"
      ],
      "Resource": "*"
    }
  ]
}
```

## 7. CI/CD Pipeline Role (GitHub Actions/CodePipeline)

### Purpose
Automated deployment and CI/CD operations.

### Role Name
`weather-etl-cicd-role`

### Trust Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT-ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR-GITHUB-USERNAME/weather-etl-platform:*"
        }
      }
    }
  ]
}
```

### Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:RegisterTaskDefinition",
        "ecs:DescribeTaskDefinition"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::weather-etl-dashboard/*",
        "arn:aws:s3:::weather-etl-dashboard"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudfront:CreateInvalidation"
      ],
      "Resource": "*"
    }
  ]
}
```

## Role Assignment Matrix

| Component | Role | Primary Permissions |
|-----------|------|-------------------|
| AWS Glue ETL Jobs | `weather-etl-glue-processing-role` | S3, Glue Catalog, CloudWatch Logs |
| Apache Flink Streaming | `weather-etl-stream-processing-role` | MSK, S3, CloudWatch Logs |
| Query Service (Athena) | `weather-etl-query-service-role` | Athena, S3, Glue Catalog |
| Microservices (ECS) | `weather-etl-application-runtime-role` | S3, MSK, SSM, Secrets Manager |
| ECS Task Execution | `weather-etl-ecs-execution-role` | ECR, CloudWatch Logs |
| React Dashboard | `weather-etl-dashboard-deployment-role` | S3, CloudFront |
| CI/CD Pipeline | `weather-etl-cicd-role` | ECR, ECS, S3, CloudFront |

## Security Best Practices

1. **Principle of Least Privilege**: Each role has only the minimum permissions required.
2. **Resource-Specific Access**: Permissions are scoped to specific resources where possible.
3. **Cross-Account Access**: Use external ID for cross-account role assumptions.
4. **Regular Auditing**: Review and audit role permissions regularly.
5. **Temporary Credentials**: Use temporary credentials with session tokens.
6. **MFA Requirements**: Consider requiring MFA for sensitive operations.

## Setup Instructions

1. **Create Roles**: Use the AWS CLI or CloudFormation to create these roles.
2. **Attach Policies**: Attach both managed and custom policies to each role.
3. **Configure Services**: Configure each AWS service to use the appropriate role.
4. **Test Permissions**: Verify that services can access required resources.
5. **Monitor Usage**: Use CloudTrail to monitor role usage and access patterns.

## Terraform Configuration

The provided Terraform configuration in `infrastructure/terraform/` automatically creates these roles with the proper permissions. To deploy:

```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

## CloudFormation Alternative

For teams preferring CloudFormation, equivalent templates are available in `infrastructure/cloudformation/` directory.

## Troubleshooting

### Common Issues

1. **Access Denied Errors**: Check role trust policies and permission boundaries.
2. **Cross-Service Access**: Ensure services can assume the correct roles.
3. **Resource Naming**: Verify resource ARNs match the actual resource names.
4. **Region Restrictions**: Some permissions may be region-specific.

### Debugging Steps

1. Check CloudTrail logs for access attempts
2. Verify role trust relationships
3. Test permissions using AWS CLI with `--assume-role`
4. Use IAM Policy Simulator for testing

## Additional Resources

- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS ETL Best Practices](https://aws.amazon.com/glue/getting-started/)
- [ECS Task Roles](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html)
- [Apache Flink on AWS](https://docs.aws.amazon.com/kinesisanalytics/latest/java/getting-started.html)
