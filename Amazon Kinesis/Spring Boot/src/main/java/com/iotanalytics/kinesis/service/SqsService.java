package com.iotanalytics.kinesis.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for SQS operations
 */
public class SqsService {

    private final Map<String, List<Object>> messageQueues;

    public SqsService() {
        this.messageQueues = new HashMap<>();
    }

    /**
     * Sends message to SQS queue
     */
    public void sendMessage(String queueName, Object message) {
        try {
            messageQueues.computeIfAbsent(queueName, k -> new ArrayList<>()).add(message);
            System.out.println("Sent message to queue: " + queueName);
        } catch (Exception e) {
            System.err.println("Error sending message to queue: " + e.getMessage());
            throw new RuntimeException("Failed to send message to SQS", e);
        }
    }

    /**
     * Receives messages from SQS queue
     */
    public List<Object> receiveMessages(String queueName) {
        List<Object> messages = messageQueues.get(queueName);
        return messages != null ? new ArrayList<>(messages) : new ArrayList<>();
    }

    /**
     * Gets queue size
     */
    public int getQueueSize(String queueName) {
        List<Object> messages = messageQueues.get(queueName);
        return messages != null ? messages.size() : 0;
    }

    /**
     * Clears all messages from queue
     */
    public void clearQueue(String queueName) {
        messageQueues.remove(queueName);
    }
}
