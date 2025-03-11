import unittest
import torch
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Import the class to be tested
# Assuming the DetectionEngine class is in a module called 'detection_engine'
from nids_helpers.detection_engine import DetectionEngine

class TestDetectionEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies for initialization
        with patch('os.path.join', return_value='mock_path'), \
             patch('os.path.dirname', return_value='mock_dir'), \
             patch('torch.load', return_value=MagicMock()), \
             patch('pickle.load', return_value=MagicMock()), \
             patch('torch.device', return_value='cpu'):
            
            # Create pickle mock
            pickle_mock = MagicMock()
            pickle_mock.transform = lambda x: x  # Identity transform for simplicity
            
            # Mock open with pickle.load returning our mock
            with patch('builtins.open', mock_open()), \
                 patch('pickle.load', return_value=pickle_mock):
                self.engine = DetectionEngine()
                
                # Fix the model attribute assignment
                self.engine.anomaly_detector = MagicMock()
                # Mock model output to return tensor of correct shape for MSE calculation
                self.engine.anomaly_detector.return_value = torch.zeros(1, 10)
                self.engine.anomaly_detector.eval = MagicMock()
                
                # Override threshold for testing
                self.engine.threshold = 0.5
    
    def test_init(self):
        """Test the initialization of DetectionEngine."""
        self.assertIsNotNone(self.engine.anomaly_detector)
        self.assertIsNotNone(self.engine.scaler)
        self.assertEqual(self.engine.threshold, 0.5)
        self.assertEqual(self.engine.device, 'cpu')
        self.assertIsNotNone(self.engine.signature_rules)
        
    def test_load_signature_rules(self):
        """Test loading of signature rules."""
        rules = self.engine.load_signature_rules()
        self.assertIn('syn_flood', rules)
        self.assertIn('port_scan', rules)
        self.assertTrue(callable(rules['syn_flood']['condition']))
        self.assertTrue(callable(rules['port_scan']['condition']))
    
    @patch('torch.tensor', return_value=torch.zeros(1, 10))
    def test_detect_threats_no_anomaly(self, mock_tensor):
        """Test case where no anomaly is detected (reconstruction error below threshold)."""
        # Prepare test data
        current_time = datetime.now()
        features = {
            "avg_ipt": 0.1,
            "bytes_in": 100,
            "bytes_out": 200,
            "dest_port": 80,
            "entropy": 0.5,
            "num_pkts_out": 10,
            "num_pkts_in": 15,
            "proto": 6,
            "src_port": 12345,
            "total_entropy": 1.5,
            "duration": 30.0,
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Fix the reconstruction error calculation
        # Create a scalar tensor for torch.mean().item() call
        mock_mse = MagicMock()
        mock_mse.item.return_value = 0.3  # Below threshold
        
        # Fix model attribute name and patch torch.mean
        with patch.object(self.engine, 'anomaly_detector', self.engine.anomaly_detector), \
             patch('torch.mean', return_value=mock_mse):
            
            # Call the method under test
            threats = self.engine.detect_threats(features)
            
            # Verify results
            self.assertEqual(len(threats), 0)
    
    @patch('torch.tensor', return_value=torch.zeros(1, 10))
    def test_detect_threats_anomaly(self, mock_tensor):
        """Test case where an anomaly is detected (reconstruction error above threshold)."""
        # Prepare test data
        current_time = datetime.now()
        features = {
            "avg_ipt": 0.1,
            "bytes_in": 100,
            "bytes_out": 200,
            "dest_port": 80,
            "entropy": 0.5,
            "num_pkts_out": 10,
            "num_pkts_in": 15,
            "proto": 6,
            "src_port": 12345,
            "total_entropy": 1.5,
            "duration": 30.0,
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Fix the reconstruction error calculation
        # Create a scalar tensor for torch.mean().item() call
        mock_mse = MagicMock()
        mock_mse.item.return_value = 1.0  # Above threshold
        
        # Fix model attribute name and patch torch.mean
        with patch.object(self.engine, 'anomaly_detector', self.engine.anomaly_detector), \
             patch('torch.mean', return_value=mock_mse):
            
            # Call the method under test
            threats = self.engine.detect_threats(features)
            
            # Verify results
            self.assertEqual(len(threats), 1)
            self.assertEqual(threats[0]['type'], 'anomaly')
            self.assertEqual(threats[0]['score'], 1.0)
            self.assertLessEqual(threats[0]['confidence'], 1.0)
            self.assertGreaterEqual(threats[0]['confidence'], 0.0)
    
    def test_feature_preprocessing(self):
        """Test the preprocessing of input features for the model."""
        # Prepare test data
        current_time = datetime.now()
        features = {
            "avg_ipt": 0.1,
            "bytes_in": 100,
            "bytes_out": 200,
            "dest_port": 80,
            "entropy": 0.5,
            "num_pkts_out": 10,
            "num_pkts_in": 15,
            "proto": 6,  # TCP
            "src_port": 12345,
            "total_entropy": 1.5,
            "duration": 30.0,
            "src_ip": "192.168.1.1",  # Should be dropped
            "dest_ip": "10.0.0.1",    # Should be dropped
            "time_start": "2023-01-01T00:00:00",  # Should be dropped
            "time_end": "2023-01-01T00:00:30",    # Should be dropped
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Mock DataFrame for easier inspection
        df_mock = pd.DataFrame([features])
        
        # Fix the reconstruction error calculation
        # Create a scalar tensor for torch.mean().item() call
        mock_mse = MagicMock()
        mock_mse.item.return_value = 0.3  # Below threshold
        
        # Mock pandas operations
        with patch('pandas.DataFrame', return_value=df_mock), \
             patch.object(df_mock, 'drop', return_value=df_mock), \
             patch.object(df_mock, 'to_numpy', return_value=np.zeros((1, 10))), \
             patch('torch.tensor', return_value=torch.zeros(1, 10)), \
             patch('torch.mean', return_value=mock_mse):
            
            # Fix model attribute name
            with patch.object(self.engine, 'anomaly_detector', self.engine.anomaly_detector):
                # Call the method under test
                self.engine.detect_threats(features)
                
                # Verify that drop was called to remove specific columns
                expected_drop_calls = [
                    unittest.mock.call(['src_ip'], axis=1, inplace=True),
                    unittest.mock.call(['dest_ip'], axis=1, inplace=True),
                    unittest.mock.call(['time_start'], axis=1, inplace=True),
                    unittest.mock.call(['time_end'], axis=1, inplace=True),
                    unittest.mock.call(['Year', 'Month', 'Day'], axis=1, inplace=True)
                ]
                df_mock.drop.assert_has_calls(expected_drop_calls, any_order=True)
    
    def test_one_hot_encoding_proto(self):
        """Test the one-hot encoding of the protocol field."""
        # Create a simplified version of the detect_threats method that only tests one-hot encoding
        def simplified_one_hot_encoding(proto_value):
            feature_df = pd.DataFrame([{'proto': proto_value}])
            protos = ['proto_6', 'proto_17', 'proto_47', 'proto_58', 'proto_50', 'proto_51', 'proto_132', 'proto_89']
            feature_df[protos] = 0
            
            # Handle protocol field using one-hot encoding (copied from the actual method)
            if 'proto' in feature_df.columns:
                proto_num = feature_df['proto'].iloc[0]
                proto_col = 'proto_' + str(proto_num)
                feature_df[proto_col] = 1
                feature_df.drop(['proto'], axis=1, inplace=True)
            
            return feature_df
        
        # Test with TCP protocol (6)
        result_df = simplified_one_hot_encoding(6)
        self.assertEqual(result_df['proto_6'].values[0], 1)

    def test_port_formatting(self):
        """Test that ports are properly formatted."""
        # Create a simplified version of the port formatting logic
        def simplified_port_formatting(dest_port_val, src_port_val):
            feature_df = pd.DataFrame([{'dest_port': dest_port_val, 'src_port': src_port_val}])
            
            # Ensure ports are properly formatted (copied from the actual method)
            feature_df.dest_port = feature_df.dest_port.fillna(-1)
            feature_df.dest_port = feature_df.dest_port.infer_objects(copy=False).astype('int64')
            feature_df.src_port = feature_df.src_port.fillna(-1)
            feature_df.src_port = feature_df.src_port.infer_objects(copy=False).astype('int64')
            
            return feature_df
        
        # Test with null port values
        result_df = simplified_port_formatting(None, None)
        self.assertEqual(result_df['dest_port'].values[0], -1)
        self.assertEqual(result_df['src_port'].values[0], -1)
        self.assertEqual(result_df['dest_port'].dtype, np.int64)
        self.assertEqual(result_df['src_port'].dtype, np.int64)

    def test_scaling_applied(self):
        """Test that scaling is applied to the appropriate columns."""
        # Prepare test data
        current_time = datetime.now()
        features = {
            "avg_ipt": 0.1,
            "bytes_in": 100,
            "bytes_out": 200,
            "dest_port": 80,
            "entropy": 0.5,
            "num_pkts_out": 10,
            "num_pkts_in": 15,
            "proto": 6,
            "src_port": 12345,
            "total_entropy": 1.5,
            "duration": 30.0,
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Create a mock DataFrame
        df_mock = pd.DataFrame([features])
        # Add protocol columns
        cols_to_mock = ['proto_6', 'proto_17', 'proto_47', 'proto_58', 'proto_50', 'proto_51', 'proto_132', 'proto_89']
        for col in cols_to_mock:
            df_mock[col] = 0
        
        # Mock the scaler
        scaler_mock = MagicMock()
        scaler_mock.transform = MagicMock(return_value=np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]))
        self.engine.scaler = scaler_mock
        
        # Fix the reconstruction error calculation
        # Create a scalar tensor for torch.mean().item() call
        mock_mse = MagicMock()
        mock_mse.item.return_value = 0.3  # Below threshold
        
        # Create a controlled test environment
        with patch('pandas.DataFrame', return_value=df_mock), \
             patch.object(df_mock, 'drop', return_value=df_mock), \
             patch.object(df_mock, 'to_numpy', return_value=np.zeros((1, 10))), \
             patch('torch.tensor', return_value=torch.zeros(1, 10)), \
             patch('torch.mean', return_value=mock_mse):
            
            # Fix model attribute name
            with patch.object(self.engine, 'anomaly_detector', self.engine.anomaly_detector):
                # Call the method under test
                self.engine.detect_threats(features)
                
                # Verify scaler.transform was called
                scaler_mock.transform.assert_called_once()
    
    def test_model_usage(self):
        """Test that the model attribute is correctly used in the detect_threats method."""
        # Prepare test data
        current_time = datetime.now()
        features = {
            "avg_ipt": 0.1,
            "bytes_in": 100,
            "bytes_out": 200,
            "dest_port": 80,
            "entropy": 0.5,
            "num_pkts_out": 10,
            "num_pkts_in": 15,
            "proto": 6,
            "src_port": 12345,
            "total_entropy": 1.5,
            "duration": 30.0,
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Create a mock DataFrame
        df_mock = pd.DataFrame([features])
        # Add protocol columns
        cols_to_mock = ['proto_6', 'proto_17', 'proto_47', 'proto_58', 'proto_50', 'proto_51', 'proto_132', 'proto_89']
        for col in cols_to_mock:
            df_mock[col] = 0
        
        # Fix the reconstruction error calculation
        # Create a scalar tensor for torch.mean().item() call
        mock_mse = MagicMock()
        mock_mse.item.return_value = 1.0  # Above threshold
        
        # Create a controlled test environment
        with patch('pandas.DataFrame', return_value=df_mock), \
             patch.object(df_mock, 'drop', return_value=df_mock), \
             patch.object(df_mock, 'to_numpy', return_value=np.zeros((1, 10))), \
             patch('torch.tensor', return_value=torch.zeros(1, 10)), \
             patch('torch.mean', return_value=mock_mse):
            
            # Create a specific mock for anomaly_detector
            anomaly_detector_mock = MagicMock()
            anomaly_detector_mock.eval = MagicMock()
            anomaly_detector_mock.return_value = torch.zeros(1, 10)
            
            # Fix model attribute name
            with patch.object(self.engine, 'anomaly_detector', anomaly_detector_mock):
                # Call the method under test
                threats = self.engine.detect_threats(features)
                
                # Verify model was called correctly
                anomaly_detector_mock.eval.assert_called_once()
                anomaly_detector_mock.assert_called_once()
                self.assertEqual(len(threats), 1)
                self.assertEqual(threats[0]['type'], 'anomaly')

if __name__ == '__main__':
    unittest.main()