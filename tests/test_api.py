import unittest
from unittest.mock import MagicMock, patch
import json
from fastapi.testclient import TestClient

# Mock the exp_hub global variable
import sys
import builtins
original_import = builtins.__import__

def mock_import(name, *args, **kwargs):
    if name == 'ExperimentHubAPI':
        # Create a mock module
        import types
        mock_module = types.ModuleType('ExperimentHubAPI')
        mock_module.exp_hub = MagicMock()
        mock_module.app = None  # Will be set in the test
        return mock_module
    return original_import(name, *args, **kwargs)

# Apply the mock
builtins.__import__ = mock_import

# Now import the module
from ExperimentHubAPI import app, exp_hub

# Restore the original import
builtins.__import__ = original_import

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Reset the mock
        exp_hub.reset_mock()
    
    def test_health_endpoint(self):
        # Configure mock
        exp_hub.check_plugin_health.return_value = {
            "mlops": {
                "healthy": True,
                "details": {"status": "connected"}
            },
            "datastream": {
                "healthy": True,
                "details": {"status": "connected"}
            }
        }
        
        # Test endpoint
        response = self.client.get("/health")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("mlops", data)
        self.assertIn("datastream", data)
        self.assertTrue(data["mlops"]["healthy"])
        self.assertTrue(data["datastream"]["healthy"])
    
    def test_available_experiments_endpoint(self):
        # Configure mock
        exp_hub.available_experiments = {
            "time_series": {
                "module": "framework.TimeSeriesAnalysis",
                "class": "TimeSeriesExperiment"
            },
            "fault_detection": {
                "module": "framework.FaultDetection",
                "class": "FaultDetectionExperiment"
            }
        }
        
        # Test endpoint
        response = self.client.get("/available_experiments")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("time_series", data)
        self.assertIn("fault_detection", data)
    
    def test_create_experiment_endpoint(self):
        # Configure mock
        exp_hub.available_experiments = {
            "time_series": {
                "module": "framework.TimeSeriesAnalysis",
                "class": "TimeSeriesExperiment"
            }
        }
        exp_hub.create_experiment.return_value = "test_experiment"
        
        # Test endpoint
        config = {"param1": "value1"}
        response = self.client.post(
            "/create_experiment/test_experiment/time_series",
            json=config
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("test_experiment", data["message"])
        self.assertIn("time_series", data["message"])
        
        # Verify create_experiment was called
        exp_hub.create_experiment.assert_called_once_with(
            "test_experiment", "time_series", config
        )
    
    def test_create_experiment_unknown_type(self):
        # Configure mock
        exp_hub.available_experiments = {
            "time_series": {
                "module": "framework.TimeSeriesAnalysis",
                "class": "TimeSeriesExperiment"
            }
        }
        
        # Test endpoint with unknown experiment type
        config = {"param1": "value1"}
        response = self.client.post(
            "/create_experiment/test_experiment/unknown_type",
            json=config
        )
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Unknown experiment type", data["detail"])
        
        # Verify create_experiment was not called
        exp_hub.create_experiment.assert_not_called()

if __name__ == '__main__':
    unittest.main()
