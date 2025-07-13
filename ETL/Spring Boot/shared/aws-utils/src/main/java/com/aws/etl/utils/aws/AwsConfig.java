package com.aws.etl.utils.aws;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.athena.AthenaClient;
import software.amazon.awssdk.services.cloudwatch.CloudWatchClient;
import software.amazon.awssdk.services.glue.GlueClient;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.secretsmanager.SecretsManagerClient;
import software.amazon.awssdk.services.ssm.SsmClient;

/**
 * AWS SDK Configuration for all AWS service clients.
 * Uses default credential provider chain and configurable region.
 */
@Configuration
public class AwsConfig {

    @Value("${aws.region:us-east-1}")
    private String awsRegion;

    @Bean
    public Region region() {
        return Region.of(awsRegion);
    }

    @Bean
    public DefaultCredentialsProvider credentialsProvider() {
        return DefaultCredentialsProvider.create();
    }

    @Bean
    public S3Client s3Client(Region region, DefaultCredentialsProvider credentialsProvider) {
        return S3Client.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }

    @Bean
    public GlueClient glueClient(Region region, DefaultCredentialsProvider credentialsProvider) {
        return GlueClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }

    @Bean
    public AthenaClient athenaClient(Region region, DefaultCredentialsProvider credentialsProvider) {
        return AthenaClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }

    @Bean
    public SsmClient ssmClient(Region region, DefaultCredentialsProvider credentialsProvider) {
        return SsmClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }

    @Bean
    public SecretsManagerClient secretsManagerClient(Region region, DefaultCredentialsProvider credentialsProvider) {
        return SecretsManagerClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }

    @Bean
    public CloudWatchClient cloudWatchClient(Region region, DefaultCredentialsProvider credentialsProvider) {
        return CloudWatchClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
                .build();
    }
}
