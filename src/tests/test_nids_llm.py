import unittest
from datetime import datetime
from unittest.mock import Mock, patch
import tempfile
import os
import sys

from nids_helpers.alert_system import AlertSystem

class TestNIDSLLMIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary log file
        self.temp_log = tempfile.NamedTemporaryFile(delete=False)
        self.alert_system = AlertSystem(
            log_file=self.temp_log.name,
            llm_endpoint="http://localhost:5000/analyze_alert"  # Mocked LLM endpoint
        )

    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_log.name)

    def test_basic_alert_generation(self):
        """Test basic alert generation without LLM integration"""
        threat = {
            'type': 'anomaly',
            'confidence': 0.9,
            'score': -0.75
        }
        
        packet_info = {
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.1',
            'source_port': 12345,
            'destination_port': 80
        }
        
        self.alert_system.generate_alert(threat, packet_info)
        
        # Read the log file and check if alert was logged
        with open(self.temp_log.name, 'r') as f:
            log_contents = f.read()
            
        self.assertIn('anomaly', log_contents)
        self.assertIn('192.168.1.100', log_contents)
        self.assertIn('High confidence threat detected', log_contents)

    @patch('requests.post')
    def test_llm_integration(self, mock_post):
        """Test LLM integration with mocked response"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "analysis": "Critical severity. Possible unauthorized access attempt.",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat()
        }
        mock_post.return_value = mock_response

        threat = {
            'type': 'port_scan',
            'confidence': 0.95,
            'score': -0.8
        }
        
        packet_info = {
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.1',
            'source_port': 12345,
            'destination_port': 80
        }
        
        self.alert_system.generate_alert(threat, packet_info)
        
        # Verify LLM endpoint was called
        mock_post.assert_called_once()
        
        # Check log file contains both alert and LLM analysis
        with open(self.temp_log.name, 'r') as f:
            log_contents = f.read()
            
        self.assertIn('port_scan', log_contents)
        self.assertIn('LLM Analysis', log_contents)

def test_live_integration(ngrok_endpoint):
    """
    Test live integration with actual LLM endpoint
    (Run this separately from unit tests)
    """
    
    # Initialize with your ngrok URL
    alert_system = AlertSystem(
        log_file="test_alerts.log",
        llm_endpoint=ngrok_endpoint + "/analyze_alert"
    )
    
    # Test cases with varying confidence levels
    test_cases = [
        {
            'threat': {
                'type': 'syn_flood',
                'confidence': 0.9,
                'score': -0.85
            },
            'packet_info': {
                'source_ip': '192.168.1.100',
                'destination_ip': '10.0.0.1',
                'source_port': 12345,
                'destination_port': 80
            }
        },
        {
            'threat': {
                'type': 'port_scan',
                'confidence': 0.6,
                'score': -0.65
            },
            'packet_info': {
                'source_ip': '192.168.1.101',
                'destination_ip': '10.0.0.2',
                'source_port': 54321,
                'destination_port': 443
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting alert for {test_case['threat']['type']}")
        alert_system.generate_alert(test_case['threat'], test_case['packet_info'])
        print("Alert generated, check test_alerts.log for results")


if __name__ == '__main__':
    # check for valid usage
    if len(sys.argv) < 2:
        print("Usage: python test_nids_llm.py <ngrok_endpoint>")
        sys.exit(1)

    ngrok_endpoint = sys.argv[1]  # Extract ngrok endpoint
    sys.argv = [sys.argv[0]]  # Reset sys.argv to prevent unittest from interpreting extra args

    # Run unit tests
    unittest.main(exit=False)
    
    # Run live integration test
    print("\nRunning live integration test...")
    test_live_integration(ngrok_endpoint)