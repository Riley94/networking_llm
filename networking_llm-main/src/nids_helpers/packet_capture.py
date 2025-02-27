import threading
import queue
import pyshark

class PacketCapture:
    def __init__(self):
        self.packet_queue = queue.Queue()
        self.stop_capture = threading.Event()
        self.capture = None

    def packet_callback(self, packet):
        if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):
            self.packet_queue.put(packet)

    def start_capture(self, interface="wlo1"):
        def capture_thread():
            self.capture = pyshark.LiveCapture(
                interface=interface,
                include_raw=True,
                use_json=True
            )
            for packet in self.capture.sniff_continuously():
                if self.stop_capture.is_set():
                    break
                if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):
                    self.packet_queue.put(packet)

        self.capture_thread = threading.Thread(target=capture_thread)
        self.capture_thread.start()

    def stop(self):
        self.stop_capture.set()
        if self.capture:
            self.capture.close()
        self.capture_thread.join()
