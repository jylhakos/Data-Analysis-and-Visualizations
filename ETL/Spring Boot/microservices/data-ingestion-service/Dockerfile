# Multi-stage Dockerfile for Spring Boot microservices
FROM openjdk:17-jdk-slim as builder

# Set working directory
WORKDIR /app

# Copy Maven files
COPY pom.xml ./
COPY .mvn .mvn
COPY mvnw ./

# Copy shared modules
COPY shared/ shared/

# Copy microservice source
COPY microservices/data-ingestion-service/ microservices/data-ingestion-service/

# Build the application
RUN ./mvnw clean package -DskipTests -pl microservices/data-ingestion-service -am

# Runtime stage
FROM openjdk:17-jre-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy the built JAR from builder stage
COPY --from=builder /app/microservices/data-ingestion-service/target/*.jar app.jar

# Create logs directory
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/actuator/health || exit 1

# Expose ports
EXPOSE 8080 8081

# Environment variables
ENV JAVA_OPTS="-Xmx512m -Xms256m" \
    SPRING_PROFILES_ACTIVE="docker" \
    SERVER_PORT=8080 \
    MANAGEMENT_SERVER_PORT=8081

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Run the application
CMD ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
