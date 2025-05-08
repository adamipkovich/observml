import pika
import time
import logging
from framework.plugins.datastream import DataStreamPlugin
from typing import Any, Dict, Tuple

class RabbitMQPlugin(DataStreamPlugin):
    """RabbitMQ implementation of DataStreamPlugin"""
    plugin_type = "datastream"
    
    def __init__(self, host: str, port: str|int, username: str, password: str, **kwargs):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
    
    def initialize(self) -> None:
        """Initialize the plugin"""
        self.connect()
    
    def shutdown(self) -> None:
        """Clean up resources"""
        self.disconnect()
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if RabbitMQ plugin is working correctly"""
        try:
            # Check if connection is open
            if self.connection is None or self.connection.is_closed:
                self.connect()
            
            # Try to create a test queue
            test_queue = f"health_check_{int(time.time())}"
            self.channel.queue_declare(queue=test_queue, durable=False)
            self.channel.queue_delete(queue=test_queue)
            
            return True, {
                "status": "connected", 
                "host": self.host, 
                "port": self.port
            }
        except Exception as e:
            return False, {
                "status": "error", 
                "message": str(e), 
                "host": self.host, 
                "port": self.port
            }
    
    def connect(self, **kwargs) -> None:
        """Connect to RabbitMQ"""
        if self.connection is not None and not self.connection.is_closed:
            return
        
        credentials = pika.PlainCredentials(username=self.username, password=self.password)
        while self.connection is None:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=self.host, 
                        port=self.port, 
                        credentials=credentials, 
                        heartbeat=0
                    )
                )
            except pika.exceptions.AMQPConnectionError:
                logging.error(f"Connection to RabbitMQ failed at {self.host}:{self.port}. Retrying...")
                time.sleep(3)
        
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
    
    def disconnect(self) -> None:
        """Disconnect from RabbitMQ"""
        if self.channel is not None and self.channel.is_open:
            self.channel.close()
            self.channel = None
        
        if self.connection is not None and self.connection.is_open:
            self.connection.close()
            self.connection = None
    
    def create_queue(self, queue_name: str) -> None:
        """Create a queue in RabbitMQ"""
        self.channel.queue_declare(queue=queue_name, durable=True)
    
    def pull_data(self, queue: str) -> Any:
        """Pull data from a queue"""
        method_frame, header_frame, body = self.channel.basic_get(queue)
        if method_frame:
            data = body.decode("utf-8")
            self.channel.basic_ack(method_frame.delivery_tag)
            return data
        else:
            return ""
    
    def flush_queue(self, queue: str) -> None:
        """Flush a queue"""
        self.channel.queue_declare(queue=queue, durable=True)
        self.channel.queue_purge(queue)
