<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for AWS ETL Microservices Project

## Project Context
This is a Spring Boot microservices ETL application designed for AWS deployment with the following key characteristics:

- **Architecture**: Microservices using Spring Boot
- **Cloud Platform**: Amazon AWS
- **ETL Framework**: AWS Glue for batch processing
- **Stream Processing**: Apache Flink for real-time data
- **Data Format**: Apache Iceberg for schema evolution
- **Query Engine**: Amazon Athena
- **Message Broker**: MQTT for weather station data
- **Frontend**: React dashboard for visualization
- **Infrastructure**: Terraform for IaC

## Code Generation Guidelines

### Spring Boot Microservices
- Use Spring Boot 3.x with Java 17+
- Implement reactive programming with Spring WebFlux where appropriate
- Use Spring Cloud for microservices patterns (Config, Discovery, Gateway)
- Include proper logging with SLF4J and Logback
- Implement health checks and metrics with Spring Actuator

### AWS Integration
- Use AWS SDK for Java v2
- Implement proper retry mechanisms and circuit breakers
- Include AWS X-Ray tracing for distributed systems
- Use AWS Parameter Store for configuration management
- Implement proper IAM role-based authentication

### Data Processing
- Follow AWS Glue job patterns for ETL operations
- Use Apache Iceberg table format specifications
- Implement Apache Flink streaming patterns
- Include proper error handling and dead letter queues
- Use Apache Kafka/MSK for message streaming

### Infrastructure as Code
- Generate Terraform configurations using HCL syntax
- Include proper variable definitions and outputs
- Implement modular Terraform structure
- Include CloudFormation templates as alternatives
- Use AWS best practices for security and networking

### Weather Station Data Model
- Temperature measurements every hour
- 5 weather stations within 100km radius
- Include metadata: station_id, timestamp, temperature, humidity, pressure
- Implement proper data validation and sanitization
- Use ISO 8601 for timestamp formatting

### Security Considerations
- Implement least privilege IAM policies
- Use VPC with proper subnet configuration
- Include encryption at rest and in transit
- Implement proper API authentication (JWT/OAuth2)
- Use AWS Secrets Manager for sensitive data

### Monitoring and Observability
- Include CloudWatch metrics and alarms
- Implement distributed tracing with X-Ray
- Use structured logging with correlation IDs
- Include proper error handling and alerting
- Implement health checks for all services

### Testing Patterns
- Use JUnit 5 and Mockito for unit testing
- Include integration tests with Testcontainers
- Implement contract testing for microservices
- Use WireMock for external service mocking
- Include performance testing considerations
