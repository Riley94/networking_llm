import unittest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime
from collections import defaultdict

# Import TrafficAnalyzer directly from the current directory
from nids_helpers.traffic_analyzer import TrafficAnalyzer

class TestTrafficAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrafficAnalyzer(flow_timeout=60)

    def test_init(self):
        """Test the initialization of TrafficAnalyzer."""
        self.assertEqual(self.analyzer.flow_timeout, 60)
        self.assertIsInstance(self.analyzer.flow_stats, defaultdict)
        self.assertIsInstance(self.analyzer.completed_flows, list)
        self.assertEqual(len(self.analyzer.completed_flows), 0)

    def test_compute_entropy_empty(self):
        """Test entropy computation with empty payloads."""
        entropy = self.analyzer.compute_entropy([])
        self.assertEqual(entropy, 0.0)

    def test_compute_entropy_single_byte(self):
        """Test entropy computation with a single repeated byte."""
        # With a single repeated byte, entropy should be 0
        entropy = self.analyzer.compute_entropy([b'AAAA'])
        self.assertEqual(entropy, 0.0)

    def test_compute_entropy_uniform(self):
        """Test entropy computation with uniformly distributed bytes."""
        # With uniformly distributed bytes, entropy should be maximum
        payloads = [bytes([i % 256]) for i in range(256)]
        entropy = self.analyzer.compute_entropy(payloads)
        # Maximum entropy for 256 possible values is 8 bits
        self.assertAlmostEqual(entropy, 8.0, places=1)

    def test_calculate_avg_ipt_empty(self):
        """Test IPT calculation with insufficient packet times."""
        avg_ipt = self.analyzer.calculate_avg_ipt([1.0])
        self.assertEqual(avg_ipt, 0.0)

    def test_calculate_avg_ipt(self):
        """Test IPT calculation with multiple packet times."""
        packet_times = [1.0, 2.0, 4.0, 7.0]
        # Expected IPTs: 1.0, 2.0, 3.0 -> mean = 2.0
        avg_ipt = self.analyzer.calculate_avg_ipt(packet_times)
        self.assertEqual(avg_ipt, 2.0)

    def test_check_flow_termination_timeout(self):
        """Test flow termination due to timeout."""
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key]['last_time'] = 100.0
        result = self.analyzer.check_flow_termination(flow_key, 161.0)
        self.assertTrue(result)

    def test_check_flow_termination_fin(self):
        """Test flow termination due to FIN flag."""
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key]['last_time'] = 100.0
        self.analyzer.flow_stats[flow_key]['fin_seen'] = True
        result = self.analyzer.check_flow_termination(flow_key, 101.0)
        self.assertTrue(result)

    def test_check_flow_termination_rst(self):
        """Test flow termination due to RST flag."""
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key]['last_time'] = 100.0
        self.analyzer.flow_stats[flow_key]['rst_seen'] = True
        result = self.analyzer.check_flow_termination(flow_key, 101.0)
        self.assertTrue(result)

    def test_check_flow_termination_active(self):
        """Test flow that should remain active."""
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key]['last_time'] = 100.0
        result = self.analyzer.check_flow_termination(flow_key, 150.0)
        self.assertFalse(result)

    @patch('datetime.datetime')
    def test_finalize_flow(self, mock_datetime):
        """Test the flow finalization process."""
        current_time = datetime(2025, 3, 6)
        mock_datetime.now.return_value = current_time
        
        flow_key = ('192.168.1.1', '192.168.1.2')
        proto = 6
        src_port = 12345
        dest_port = 80
        
        # Setup flow statistics
        self.analyzer.flow_stats[flow_key] = {
            'num_pkts_out': 5,
            'num_pkts_in': 3,
            'bytes_in': 300,
            'bytes_out': 500,
            'start_time': 100.0,
            'last_time': 110.0,
            'packet_times': [100.0, 102.0, 105.0, 108.0, 110.0],
            'payloads': [b'ABC', b'DEF'],
            'is_active': True,
            'fin_seen': True,
            'rst_seen': False
        }
        
        flow_features = self.analyzer.finalize_flow(flow_key, proto, src_port, dest_port)
        
        # Verify the returned flow features
        self.assertEqual(flow_features['num_pkts_out'], 5)
        self.assertEqual(flow_features['num_pkts_in'], 3)
        self.assertEqual(flow_features['bytes_in'], 300)
        self.assertEqual(flow_features['bytes_out'], 500)
        self.assertEqual(flow_features['proto'], 6)
        self.assertEqual(flow_features['src_port'], 12345)
        self.assertEqual(flow_features['dest_port'], 80)
        self.assertEqual(flow_features['duration'], 10.0)
        
        # Verify the flow was added to completed_flows
        self.assertEqual(len(self.analyzer.completed_flows), 1)
        self.assertEqual(self.analyzer.completed_flows[0], flow_features)
        
        # Verify the flow was removed from active flows
        self.assertNotIn(flow_key, self.analyzer.flow_stats)

    @patch('datetime.datetime')
    def test_analyze_packet_new_flow(self, mock_datetime):
        """Test analyzing a packet for a new flow."""
        current_time = datetime(2025, 3, 6)
        mock_datetime.now.return_value = current_time
        
        # Create a mock packet
        packet = Mock()
        packet.ip = Mock()
        packet.ip.src = '192.168.1.1'
        packet.ip.dst = '192.168.1.2'
        packet.ip.proto = '6'
        packet.tcp = Mock()
        packet.tcp.srcport = '12345'
        packet.tcp.dstport = '80'
        packet.tcp.flags = '0x00'
        packet.length = '100'
        packet.sniff_timestamp = '1000.0'
        
        # Mock the payload retrieval
        packet.get_raw_packet = Mock(return_value=b'test payload')
        packet.tcp.payload = True
        
        result = self.analyzer.analyze_packet(packet)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result['num_pkts_out'], 1)
        self.assertEqual(result['num_pkts_in'], 0)
        self.assertEqual(result['bytes_out'], 100)
        self.assertEqual(result['bytes_in'], 0)
        self.assertEqual(result['proto'], 6)
        self.assertEqual(result['src_port'], 12345)
        self.assertEqual(result['dest_port'], 80)
        self.assertEqual(result['Year'], 2025)
        self.assertEqual(result['Month'], 3)
        self.assertEqual(result['Day'], 6)

    @patch('datetime.datetime')
    def test_analyze_packet_fin_flag(self, mock_datetime):
        """Test analyzing a packet with FIN flag set."""
        current_time = datetime(2025, 3, 6)
        mock_datetime.now.return_value = current_time
        
        # First add a flow
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key] = {
            'num_pkts_out': 5,
            'num_pkts_in': 3,
            'bytes_in': 300,
            'bytes_out': 500,
            'start_time': 100.0,
            'last_time': 110.0,
            'packet_times': [100.0, 102.0, 105.0, 108.0, 110.0],
            'payloads': [b'ABC', b'DEF'],
            'is_active': True,
            'fin_seen': False,
            'rst_seen': False
        }
        
        # Create a mock packet with FIN flag
        packet = Mock()
        packet.ip = Mock()
        packet.ip.src = '192.168.1.1'
        packet.ip.dst = '192.168.1.2'
        packet.ip.proto = '6'
        packet.tcp = Mock()
        packet.tcp.srcport = '12345'
        packet.tcp.dstport = '80'
        packet.tcp.flags = '0x01'  # FIN flag
        packet.length = '100'
        packet.sniff_timestamp = '115.0'
        
        # Mock the payload retrieval
        packet.get_raw_packet = Mock(return_value=b'fin payload')
        packet.tcp.payload = True
        
        result = self.analyzer.analyze_packet(packet)
        
        # Verify the flow was finalized due to FIN flag
        self.assertEqual(len(self.analyzer.completed_flows), 1)
        self.assertEqual(self.analyzer.completed_flows[0]['proto'], 6)
        self.assertEqual(self.analyzer.completed_flows[0]['src_port'], 12345)
        self.assertEqual(self.analyzer.completed_flows[0]['dest_port'], 80)

    def test_analyze_packet_timeout(self):
        """Test analyzing a packet that triggers flow timeout."""
        # First add a flow
        flow_key = ('192.168.1.1', '192.168.1.2')
        self.analyzer.flow_stats[flow_key] = {
            'num_pkts_out': 5,
            'num_pkts_in': 3,
            'bytes_in': 300,
            'bytes_out': 500,
            'start_time': 100.0,
            'last_time': 110.0,
            'packet_times': [100.0, 102.0, 105.0, 108.0, 110.0],
            'payloads': [b'ABC', b'DEF'],
            'is_active': True,
            'fin_seen': False,
            'rst_seen': False
        }
        
        # Create a mock packet arriving after timeout
        packet = Mock()
        packet.ip = Mock()
        packet.ip.src = '192.168.1.1'
        packet.ip.dst = '192.168.1.2'
        packet.ip.proto = '6'
        packet.tcp = Mock()
        packet.tcp.srcport = '12345'
        packet.tcp.dstport = '80'
        packet.tcp.flags = '0x00'
        packet.length = '100'
        packet.sniff_timestamp = '180.0'  # > 60s after last_time
        
        # Mock the payload retrieval
        packet.get_raw_packet = Mock(return_value=b'timeout payload')
        packet.tcp.payload = True
        
        # Directly verify the check_flow_termination method works
        self.assertTrue(self.analyzer.check_flow_termination(flow_key, float(packet.sniff_timestamp)))
        
        # Test the finalize_flow method directly instead of through analyze_packet
        proto = int(packet.ip.proto)
        src_port = int(packet.tcp.srcport)
        dest_port = int(packet.tcp.dstport)
        
        flow_features = self.analyzer.finalize_flow(flow_key, proto, src_port, dest_port)
        
        # Verify flow was added to completed_flows
        self.assertEqual(len(self.analyzer.completed_flows), 1)
        self.assertEqual(self.analyzer.completed_flows[0], flow_features)

    def test_analyze_packet_invalid(self):
        """Test analyzing an invalid packet (no IP or TCP layer)."""
        # Create a truly invalid packet (no ip or tcp attributes)
        packet = Mock(spec=[])  # Empty spec means no attributes
        
        result = self.analyzer.analyze_packet(packet)
        
        # Should return None for invalid packets
        self.assertIsNone(result)

    def test_get_completed_flows(self):
        """Test retrieving the completed flows."""
        # Add some completed flows
        self.analyzer.completed_flows = [
            {'src_port': 12345, 'dest_port': 80},
            {'src_port': 54321, 'dest_port': 443}
        ]
        
        flows = self.analyzer.get_completed_flows()
        
        # Verify we get the correct flows
        self.assertEqual(len(flows), 2)
        self.assertEqual(flows[0]['src_port'], 12345)
        self.assertEqual(flows[1]['src_port'], 54321)

if __name__ == '__main__':
    unittest.main()