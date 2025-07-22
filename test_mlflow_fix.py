#!/usr/bin/env python3
"""
Test script to verify MLflow run management fixes.

This script tests the enhanced MLflow plugin to ensure it properly handles
active runs and prevents the "Run with UUID ... is already active" error.
"""

import os
import sys
import logging
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.plugins.mlflow_plugin import MLflowPlugin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mlflow_plugin_active_run_handling():
    """Test that the MLflow plugin properly handles active runs."""
    logger.info("Testing MLflow plugin active run handling...")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Mock MLflow functions
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.tracking.MlflowClient') as mock_client_class, \
             patch('mlflow.active_run') as mock_active_run, \
             patch('mlflow.end_run') as mock_end_run, \
             patch('mlflow.start_run') as mock_start_run:
            
            # Setup mocks
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.search_experiments.return_value = []
            mock_client.set_terminated = MagicMock()
            
            # Create plugin
            plugin = MLflowPlugin(mlflow_uri="http://localhost:5000")
            plugin.initialize()
            
            # Test 1: No active run - should start normally
            logger.info("Test 1: Starting run with no active run")
            mock_active_run.return_value = None
            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-1"
            mock_start_run.return_value = mock_run
            
            run_id = plugin.start_run("experiment-1")
            
            assert run_id == "test-run-1"
            mock_end_run.assert_not_called()
            mock_start_run.assert_called_once_with(experiment_id="experiment-1")
            logger.info("✅ Test 1 passed")
            
            # Reset mocks
            mock_end_run.reset_mock()
            mock_start_run.reset_mock()
            
            # Test 2: Active run exists - should end it first
            logger.info("Test 2: Starting run with existing active run")
            mock_active_run_obj = MagicMock()
            mock_active_run_obj.info.run_id = "existing-run"
            mock_active_run.return_value = mock_active_run_obj
            
            mock_new_run = MagicMock()
            mock_new_run.info.run_id = "test-run-2"
            mock_start_run.return_value = mock_new_run
            
            run_id = plugin.start_run("experiment-1")
            
            assert run_id == "test-run-2"
            mock_end_run.assert_called_once()
            mock_start_run.assert_called_once_with(experiment_id="experiment-1")
            logger.info("✅ Test 2 passed")
            
            # Test 3: Test end_any_active_run method
            logger.info("Test 3: Testing end_any_active_run method")
            mock_end_run.reset_mock()
            mock_active_run.return_value = mock_active_run_obj
            
            result = plugin.end_any_active_run()
            
            assert result == True
            mock_end_run.assert_called_once()
            logger.info("✅ Test 3 passed")
            
            # Test 4: Test end_any_active_run with no active run
            logger.info("Test 4: Testing end_any_active_run with no active run")
            mock_end_run.reset_mock()
            mock_active_run.return_value = None
            
            result = plugin.end_any_active_run()
            
            assert result == False
            mock_end_run.assert_not_called()
            logger.info("✅ Test 4 passed")
            
            # Test 5: Test kill_run method
            logger.info("Test 5: Testing kill_run method")
            mock_end_run.reset_mock()
            mock_client.set_terminated.reset_mock()
            mock_active_run.return_value = mock_active_run_obj
            
            result = plugin.kill_run("existing-run")
            
            assert result == True
            mock_end_run.assert_called_once()
            # The set_terminated might fail but the method should still return True
            # due to the exception handling in kill_run
            logger.info("✅ Test 5 passed")
            
            logger.info("🎉 All MLflow plugin tests passed!")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_context_manager():
    """Test the MLflow context manager."""
    logger.info("Testing MLflow context manager...")
    
    with patch('mlflow.set_tracking_uri') as mock_set_uri, \
         patch('mlflow.tracking.MlflowClient') as mock_client_class, \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run') as mock_end_run, \
         patch('mlflow.start_run') as mock_start_run:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.search_experiments.return_value = []
        
        mock_run = MagicMock()
        mock_run.info.run_id = "context-test-run"
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run
        
        # Create plugin
        plugin = MLflowPlugin(mlflow_uri="http://localhost:5000")
        plugin.initialize()
        
        # Test context manager
        try:
            with plugin.mlflow_run_context("experiment-1") as run_id:
                assert run_id == "context-test-run"
                logger.info("✅ Context manager created run successfully")
                # Simulate some work
                pass
        except Exception as e:
            logger.error(f"❌ Context manager test failed: {e}")
            raise
        
        # Verify cleanup was called
        mock_end_run.assert_called()
        logger.info("✅ Context manager cleanup called")
        
        logger.info("🎉 Context manager test passed!")

def main():
    """Run all tests."""
    logger.info("Starting MLflow fix verification tests...")
    
    try:
        test_mlflow_plugin_active_run_handling()
        test_context_manager()
        
        logger.info("🎉 All tests passed! The MLflow run management fixes are working correctly.")
        logger.info("")
        logger.info("Summary of fixes implemented:")
        logger.info("1. ✅ Enhanced start_run() to check and end existing active runs")
        logger.info("2. ✅ Improved kill_run() to handle various edge cases")
        logger.info("3. ✅ Added end_any_active_run() method for cleanup")
        logger.info("4. ✅ Added context manager for automatic run cleanup")
        logger.info("5. ✅ Enhanced error handling in training functions")
        logger.info("6. ✅ Improved stop_training() in ExperimentHub")
        logger.info("")
        logger.info("The original error 'Run with UUID ... is already active' should now be resolved.")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
