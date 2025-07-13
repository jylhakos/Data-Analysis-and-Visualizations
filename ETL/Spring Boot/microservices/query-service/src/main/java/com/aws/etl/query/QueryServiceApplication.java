package com.aws.etl.query;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Query Service Application
 * 
 * This microservice handles:
 * - Amazon Athena query execution
 * - Data lake querying
 * - Schema evolution with Apache Iceberg
 * - Query optimization and caching
 */
@SpringBootApplication(scanBasePackages = {"com.aws.etl.query", "com.aws.etl.utils"})
public class QueryServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(QueryServiceApplication.class, args);
    }
}
