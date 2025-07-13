package com.aws.etl.dashboard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Dashboard API Service Application
 * 
 * This microservice handles:
 * - REST API for dashboard frontend
 * - WebSocket real-time updates
 * - Data aggregation and caching
 * - API security and rate limiting
 */
@SpringBootApplication(scanBasePackages = {"com.aws.etl.dashboard", "com.aws.etl.utils"})
public class DashboardApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(DashboardApiApplication.class, args);
    }
}
