FROM openjdk:17-jdk-slim

WORKDIR /app

# Copy the JAR file
COPY target/kinesis-iot-processor-1.0.0.jar app.jar

# Expose the port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/actuator/health || exit 1

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]
