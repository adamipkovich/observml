import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import yaml

from framework.plugins.mlflow_plugin import MLflowPlugin
from framework.plugins.rabbitmq_plugin import RabbitMQPlugin
from framework.plugins.celery_plugin import CeleryPlugin

class TestMLflowPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = MLflowPlugin(mlflow_uri="http://localhost:5000")
        
    @patch('mlflow.tracking.MlflowClient')
    def test_initialize(self, mock_client):
        self.plugin.initialize()
        self.assertIsNotNone(self.plugin.client)
        
    @patch('mlflow.tracking.MlflowClient')
    def test_health_check_success(self, mock_client):
        mock_client.return_value.list_experiments.return_value = []
        self.plugin.initialize()
        is_healthy, details = self.plugin.health_check()
        self.assertTrue(is_healthy)
        self.assertEqual(details["status"], "connected")
        
    @patch('mlflow.tracking.MlflowClient')
    def test_health_check_failure(self, mock_client):
        mock_client.return_value.list_experiments.side_effect = Exception("Connection error")
        self.plugin.initialize()
        is_healthy, details = self.plugin.health_check()
        self.assertFalse(is_healthy)
        self.assertEqual(details["status"], "error")

class TestRabbitMQPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = RabbitMQPlugin(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
    
    @patch('pika.BlockingConnection')
    def test_initialize(self, mock_connection):
        self.plugin.initialize()
        self.assertIsNotNone(self.plugin.connection)
        self.assertIsNotNone(self.plugin.channel)
        
    @patch('pika.BlockingConnection')
    def test_health_check_success(self, mock_connection):
        self.plugin.initialize()
        is_healthy, details = self.plugin.health_check()
        self.assertTrue(is_healthy)
        self.assertEqual(details["status"], "connected")
        
    @patch('pika.BlockingConnection')
    def test_health_check_failure(self, mock_connection):
        mock_connection.side_effect = Exception("Connection error")
        is_healthy, details = self.plugin.health_check()
        self.assertFalse(is_healthy)
        self.assertEqual(details["status"], "error")

class TestCeleryPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = CeleryPlugin(
            broker_url="amqp://guest:guest@localhost:5672//",
            backend_url="redis://localhost:6379/0"
        )
    
    @patch('celery.Celery')
    def test_initialize(self, mock_celery):
        self.plugin.initialize()
        self.assertIsNotNone(self.plugin.app)
        
    @patch('celery.Celery')
    def test_health_check_success(self, mock_celery):
        mock_app = MagicMock()
        mock_app.control.ping.return_value = [{'worker1': 'pong'}]
        mock_celery.return_value = mock_app
        self.plugin.initialize()
        is_healthy, details = self.plugin.health_check()
        self.assertTrue(is_healthy)
        self.assertEqual(details["status"], "connected")
        
    @patch('celery.Celery')
    def test_health_check_failure(self, mock_celery):
        mock_app = MagicMock()
        mock_app.control.ping.return_value = []
        mock_celery.return_value = mock_app
        self.plugin.initialize()
        is_healthy, details = self.plugin.health_check()
        self.assertFalse(is_healthy)
        self.assertEqual(details["status"], "error")

if __name__ == '__main__':
    unittest.main()
