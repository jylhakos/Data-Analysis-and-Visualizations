# Apache Kafka

## Components

**Brokers**

Apache Kafka is built on a cluster of brokers (servers) that store and manage the data. Each broker acts as a node in the distributed system.

**Topics**

Data is organized into topics, which are similar to folders or categories. Producers publish messages to specific topics, and consumers subscribe to these topics to receive the data.

**Partitions**

Each topic is divided into partitions, which are replicated across multiple brokers to ensure data redundancy and fault tolerance.

**Producers**

Applications or systems that write data (messages) to Kafka topics.

**Consumers**

Applications or systems that read data (messages) from Kafka topics. 

## Event streaming

An event records the fact that "something happened" in the world or in your business.

When you read or write data to Kafka, you do this in the form of events.

Apache Kafka combines three key capabilities to implement your use cases for event streaming.

To publish (write) and subscribe to (read) streams of events, including continuous import/export of your data from other systems.

To store streams of events durably and reliably for as long as you want.

To process streams of events as they occur or retrospectively.

Topics are partitioned, showing that a topic is distributed across a number of "buckets" located on different Apache Kafka brokers.

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/Apache%20Kafka/kafka_event_streaming.png?raw=true)

*Figure: The topic has four partitions P1 to P4. Two different producer clients are publishing new events to the topic by writing events over the network to the topic's partitions. Events with the same key (denoted by their color in the figure) are written to the same partition.*

### References

[Apache Kafka](https://kafka.apache.org/)
