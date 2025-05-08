import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import yaml
import json

from framework.ExperimentHub import ExperimentHub

class TestExperimentHub(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        self.config = {
            "version": "1.0",
            "global": {
                "log_level": "info",
                "environment": "test"
            },
            "plugins": {
                "mlops": {
                    "enabled": True,
                    "type": "mlflow",
                    "config": {
                        "mlflow_uri": "http://localhost:5000"
                    }
                },
                "datastream": {
                    "enabled": True,
                    "type": "rabbitmq",
                    "config": {
                        "host": "localhost",
                        "port": 5672,
                        "username": "guest",
                        "password": "guest"
                    }
                },
                "taskqueue": {
                    "enabled": False,
                    "type": "celery",
                    "config": {
                        "broker_url": "amqp://guest:guest@localhost:5672//",
                        "backend_url": "redis://localhost:6379/0",
                        "app_name": "experiment_hub"
                    }
                }
            },
            "experiments": [
                {
                    "name": "time_series",
                    "module": "framework.TimeSeriesAnalysis",
                    "class": "TimeSeriesExperiment",
                    "enabled": True
                },
                {
                    "name": "fault_detection",
                    "module": "framework.FaultDetection",
                    "class": "FaultDetectionExperiment",
                    "enabled": True
                }
            ]
        }
        
        self.config_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        yaml.dump(self.config, self.config_file)
        self.config_file.close()
        
    def tearDown(self):
        os.unlink(self.config_file.name)
    
    @patch('framework.plugins.mlflow_plugin.MLflowPlugin')
    @patch('framework.plugins.rabbitmq_plugin.RabbitMQPlugin')
    def test_from_config(self, mock_rabbitmq, mock_mlflow):
        # Configure mocks
        mock_mlflow_instance = mock_mlflow.return_value
        mock_mlflow_instance.plugin_type = "mlops"
        
        mock_rabbitmq_instance = mock_rabbitmq.return_value
        mock_rabbitmq_instance.plugin_type = "datastream"
        
        # Test loading from config file
        hub = ExperimentHub.from_config(self.config_file.name)
        
        # Verify plugins were initialized
        self.assertIn("mlops", hub.plugins)
        self.assertIn("datastream", hub.plugins)
        self.assertNotIn("taskqueue", hub.plugins)
        
        # Verify experiments were registered
        self.assertIn("time_series", hub.available_experiments)
        self.assertIn("fault_detection", hub.available_experiments)
    
    @patch('framework.plugins.mlflow_plugin.MLflowPlugin')
    @patch('framework.plugins.rabbitmq_plugin.RabbitMQPlugin')
    def test_check_plugin_health(self, mock_rabbitmq, mock_mlflow):
        # Configure mocks
        mock_mlflow_instance = mock_mlflow.return_value
        mock_mlflow_instance.plugin_type = "mlops"
        mock_mlflow_instance.health_check.return_value = (True, {"status": "connected"})
        
        mock_rabbitmq_instance = mock_rabbitmq.return_value
        mock_rabbitmq_instance.plugin_type = "datastream"
        mock_rabbitmq_instance.health_check.return_value = (True, {"status": "connected"})
        
        # Create hub and register plugins
        hub = ExperimentHub()
        hub.register_plugin(mock_mlflow_instance)
        hub.register_plugin(mock_rabbitmq_instance)
        
        # Check health
        health_info = hub.check_plugin_health()
        
        # Verify health info
        self.assertIn("mlops", health_info)
        self.assertIn("datastream", health_info)
        self.assertTrue(health_info["mlops"]["healthy"])
        self.assertTrue(health_info["datastream"]["healthy"])
    
    @patch('framework.plugins.mlflow_plugin.MLflowPlugin')
    @patch('framework.plugins.rabbitmq_plugin.RabbitMQPlugin')
    def test_create_experiment(self, mock_rabbitmq, mock_mlflow):
        # Configure mocks
        mock_mlflow_instance = mock_mlflow.return_value
        mock_mlflow_instance.plugin_type = "mlops"
        
        mock_rabbitmq_instance = mock_rabbitmq.return_value
        mock_rabbitmq_instance.plugin_type = "datastream"
        
        # Create hub and register plugins
        hub = ExperimentHub()
        hub.register_plugin(mock_mlflow_instance)
        hub.register_plugin(mock_rabbitmq_instance)
        
        # Register available experiments
        hub.available_experiments = {
            "time_series": {
                "module": "framework.TimeSeriesAnalysis",
                "class": "TimeSeriesExperiment"
            }
        }
        
        # Mock train method
        hub.train = MagicMock(return_value=True)
        
        # Create experiment
        config = {"param1": "value1"}
        result = hub.create_experiment("test_experiment", "time_series", config)
        
        # Verify experiment was created
        self.assertEqual(result, "test_experiment")
        hub.train.assert_called_once()
        
        # Verify load_object was added to config
        train_args = hub.train.call_args[0]
        self.assertEqual(train_args[0], "test_experiment")
        self.assertIn("load_object", train_args[1])
        self.assertEqual(train_args[1]["load_object"]["module"], "framework.TimeSeriesAnalysis")
        self.assertEqual(train_args[1]["load_object"]["name"], "TimeSeriesExperiment")

if __name__ == '__main__':
    unittest.main()
