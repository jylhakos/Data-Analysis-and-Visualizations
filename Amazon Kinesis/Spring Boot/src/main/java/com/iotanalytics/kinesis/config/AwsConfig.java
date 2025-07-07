package com.iotanalytics.kinesis.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.kinesis.KinesisClient;
import software.amazon.awssdk.services.sqs.SqsClient;

import java.net.URI;

/**
 * AWS Configuration for Kinesis, DynamoDB, and SQS clients
 */
@Configuration
public class AwsConfig {

    @Value("${aws.region:us-east-1}")
    private String awsRegion;

    @Value("${aws.endpoint-override:}")
    private String endpointOverride;

    @Bean
    public KinesisClient kinesisClient() {
        var builder = KinesisClient.builder()
                .region(Region.of(awsRegion))
                .credentialsProvider(DefaultCredentialsProvider.create());
        
        if (!endpointOverride.isEmpty()) {
            builder.endpointOverride(URI.create(endpointOverride));
        }
        
        return builder.build();
    }

    @Bean
    public DynamoDbClient dynamoDbClient() {
        var builder = DynamoDbClient.builder()
                .region(Region.of(awsRegion))
                .credentialsProvider(DefaultCredentialsProvider.create());
        
        if (!endpointOverride.isEmpty()) {
            builder.endpointOverride(URI.create(endpointOverride));
        }
        
        return builder.build();
    }

    @Bean
    public SqsClient sqsClient() {
        var builder = SqsClient.builder()
                .region(Region.of(awsRegion))
                .credentialsProvider(DefaultCredentialsProvider.create());
        
        if (!endpointOverride.isEmpty()) {
            builder.endpointOverride(URI.create(endpointOverride));
        }
        
        return builder.build();
    }
}
