import unittest
from unittest.mock import MagicMock, patch
from nids_helpers.packet_capture import PacketCapture
import queue

class TestPacketCapture(unittest.TestCase):
    @patch('nids_helpers.packet_capture.pyshark.LiveCapture')
    def test_packet_capture(self, MockLiveCapture):
        # Mock packet with required attributes
        mock_packet = MagicMock()
        mock_packet.ip.src = '192.168.1.1'
        mock_packet.ip.dst = '192.168.1.2'
        mock_packet.tcp.srcport = '12345'
        mock_packet.tcp.dstport = '80'

        # Configure the mock capture to yield mock packets
        MockLiveCapture.return_value.sniff_continuously.return_value = [mock_packet]

        # Initialize PacketCapture and start capture on a mock interface
        packet_capture = PacketCapture()
        with patch.object(packet_capture, 'stop_capture', create=True) as stop_event:
            stop_event.is_set.side_effect = [False, True]  # Allow capture of one packet
            packet_capture.start_capture('mock_interface')
            packet_capture.capture_thread.join()

        # Test if packet was added to the queue
        self.assertFalse(packet_capture.packet_queue.empty())
        captured_packet = packet_capture.packet_queue.get()
        self.assertEqual(captured_packet.ip.src, '192.168.1.1')
        self.assertEqual(captured_packet.ip.dst, '192.168.1.2')
        self.assertEqual(captured_packet.tcp.srcport, '12345')
        self.assertEqual(captured_packet.tcp.dstport, '80')

    def test_stop_capture(self):
        packet_capture = PacketCapture()
        packet_capture.capture = MagicMock()
        packet_capture.capture_thread = MagicMock()
        
        packet_capture.stop()
        packet_capture.capture.close.assert_called_once()
        packet_capture.capture_thread.join.assert_called_once()

if __name__ == '__main__':
    unittest.main()