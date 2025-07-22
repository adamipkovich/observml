import unittest
from unittest.mock import MagicMock, patch


from framework.plugins.mlflow_plugin import MLflowPlugin
from framework.plugins.rabbitmq_plugin import RabbitMQPlugin

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


if __name__ == '__main__':
    unittest.main()
