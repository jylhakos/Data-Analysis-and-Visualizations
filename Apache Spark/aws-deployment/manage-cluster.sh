#!/bin/bash

CLUSTER_ID=${1}
OPERATION=${2:-"status"}

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [status|logs|terminate|ssh]"
    echo ""
    echo "Examples:"
    echo "  $0 j-XXXXXXXXX status     # Check cluster status"
    echo "  $0 j-XXXXXXXXX logs       # View cluster logs"
    echo "  $0 j-XXXXXXXXX terminate  # Terminate cluster"
    echo "  $0 j-XXXXXXXXX ssh        # SSH to master node"
    exit 1
fi

case $OPERATION in
    "status")
        echo "🔍 Checking EMR cluster status..."
        aws emr describe-cluster --cluster-id $CLUSTER_ID --query 'Cluster.{Name:Name,State:Status.State,CreationTime:Status.Timeline.CreationDateTime,Reason:Status.StateChangeReason.Message}' --output table
        
        echo ""
        echo "📊 Instance details:"
        aws emr list-instances --cluster-id $CLUSTER_ID --query 'Instances[*].{Type:InstanceGroupType,InstanceType:Ec2InstanceId,State:Status.State,PrivateIP:PrivateIpAddress,PublicIP:PublicIpAddress}' --output table
        ;;
        
    "logs")
        echo "📋 Viewing EMR cluster logs..."
        aws logs describe-log-groups --log-group-name-prefix "/aws/emr" --query 'logGroups[*].logGroupName' --output table
        
        echo ""
        echo "Recent log events:"
        aws logs describe-log-streams --log-group-name "/aws/emr/$CLUSTER_ID" --order-by LastEventTime --descending --max-items 5 --query 'logStreams[*].{StreamName:logStreamName,LastEvent:lastEventTime}' --output table
        ;;
        
    "terminate")
        echo "⚠️  WARNING: This will terminate the EMR cluster!"
        read -p "Are you sure you want to terminate cluster $CLUSTER_ID? (yes/no): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo "🛑 Terminating EMR cluster..."
            aws emr terminate-clusters --cluster-ids $CLUSTER_ID
            echo "✅ Termination request sent. Cluster will shut down in a few minutes."
        else
            echo "❌ Termination cancelled."
        fi
        ;;
        
    "ssh")
        echo "🔐 Getting SSH information..."
        MASTER_DNS=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --query 'Cluster.MasterPublicDnsName' --output text)
        
        if [ "$MASTER_DNS" = "None" ] || [ -z "$MASTER_DNS" ]; then
            echo "❌ Master node not accessible or cluster not running"
            exit 1
        fi
        
        echo "Master node: $MASTER_DNS"
        echo "SSH command: ssh -i ~/.ssh/id_rsa hadoop@$MASTER_DNS"
        echo ""
        read -p "Connect now? (y/n): " connect
        
        if [ "$connect" = "y" ]; then
            ssh -i ~/.ssh/id_rsa hadoop@$MASTER_DNS
        fi
        ;;
        
    "apps")
        echo "📱 Checking application status..."
        MASTER_DNS=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --query 'Cluster.MasterPublicDnsName' --output text)
        
        if [ "$MASTER_DNS" != "None" ] && [ -n "$MASTER_DNS" ]; then
            echo "🌐 Application URLs:"
            echo "• Spark History Server: http://$MASTER_DNS:18080"
            echo "• Hadoop NameNode: http://$MASTER_DNS:9870"
            echo "• Hadoop ResourceManager: http://$MASTER_DNS:8088"
            echo "• JupyterHub: https://$MASTER_DNS:9443"
            echo "• Zeppelin: http://$MASTER_DNS:8890"
        else
            echo "❌ Master node not accessible"
        fi
        ;;
        
    "cost")
        echo "💰 Estimating cluster costs..."
        
        # Get cluster details
        CLUSTER_INFO=$(aws emr describe-cluster --cluster-id $CLUSTER_ID)
        CREATION_TIME=$(echo $CLUSTER_INFO | jq -r '.Cluster.Status.Timeline.CreationDateTime')
        CURRENT_TIME=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)
        
        # Calculate running time
        CREATION_TIMESTAMP=$(date -d "$CREATION_TIME" +%s)
        CURRENT_TIMESTAMP=$(date +%s)
        RUNNING_HOURS=$(( (CURRENT_TIMESTAMP - CREATION_TIMESTAMP) / 3600 ))
        
        echo "Cluster running time: $RUNNING_HOURS hours"
        echo "Creation time: $CREATION_TIME"
        
        # Get instance details for cost estimation
        aws emr list-instances --cluster-id $CLUSTER_ID --query 'Instances[*].{Type:InstanceGroupType,InstanceType:InstanceType,State:Status.State}' --output table
        
        echo ""
        echo "💡 Cost estimation notes:"
        echo "• Check AWS Billing Dashboard for exact costs"
        echo "• Consider using Spot instances for core nodes to reduce costs"
        echo "• Set up auto-termination to avoid unnecessary charges"
        ;;
        
    *)
        echo "❌ Unknown operation: $OPERATION"
        echo "Valid operations: status, logs, terminate, ssh, apps, cost"
        exit 1
        ;;
esac
