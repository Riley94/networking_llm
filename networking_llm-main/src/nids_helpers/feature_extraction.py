import pyshark.tshark
import pyshark.tshark.tshark
import pyshark
import pandas as pd
from collections import defaultdict, Counter
import math
from tqdm import tqdm
import signal
import time
from contextlib import contextmanager
from threading import Thread, Event

# Function to compute Shannon entropy of payload
def compute_entropy(payload):
    if not payload:
        return 0.0

    # Count occurrences of each byte value (0-255)
    byte_counts = Counter(payload)
    total_bytes = len(payload)

    # Calculate entropy using Shannon's formula
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) 
                   for count in byte_counts.values())

    return entropy

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Set up the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class PacketCounter:
    def __init__(self, target_count):
        self.count = 0
        self.target_count = target_count
        self.start_time = time.time()
        self._stop_event = Event()
        self._monitor_thread = None

    def increment(self):
        self.count += 1
        return self.count >= self.target_count

    def _monitor_progress(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            print(f"\rPackets captured: {self.count}/{self.target_count} | Time elapsed: {elapsed:.2f}s", end='')
            time.sleep(0.5)
        print()  # New line after monitoring stops

    def start_monitoring(self):
        self._monitor_thread = Thread(target=self._monitor_progress)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

def extract_features():
    TARGET_PACKETS = 100
    session_data = defaultdict(lambda: {"num_pkts_out": 0, "num_pkts_in": 0, "total_bytes_in": 0, "total_bytes_out": 0,
                                      "entropy_values": [], "time_start": None, "time_end": None})

    interface = "wlo1"
    capture = pyshark.LiveCapture(interface=interface, use_json=True, include_raw=True)
    
    print(f"Starting capture on {interface}...")
    print(f"Target packet count: {TARGET_PACKETS}")
    
    packets = []
    counter = PacketCounter(TARGET_PACKETS)
    
    try:
        # Start the monitoring in a separate thread
        counter.start_monitoring()
        
        def capture_packets():
            for packet in capture.sniff_continuously():
                packets.append(packet)
                if counter.increment():
                    return  # Stop when we reach target count
                
        # Use a strict timeout of 30 seconds
        with timeout_context(30):
            capture_thread = Thread(target=capture_packets)
            capture_thread.daemon = True
            capture_thread.start()
            
            # Wait for either completion or timeout
            while capture_thread.is_alive() and len(packets) < TARGET_PACKETS:
                capture_thread.join(timeout=0.5)
                if len(packets) >= TARGET_PACKETS:
                    break
            
    except TimeoutException as e:
        print(f"\nCapture stopped due to timeout: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error during capture: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        counter.stop_monitoring()
        capture.close()
        print(f"Capture completed. Total packets captured: {len(packets)}")
        if len(packets) < TARGET_PACKETS:
            print(f"Warning: Only captured {len(packets)}/{TARGET_PACKETS} packets")
            print("This might be due to:")
            print("1. Low network activity on the interface")
            print("2. Insufficient permissions to capture packets")
            print("3. Interface not in promiscuous mode")
            print(f"Last packet timestamp: {packets[-1].sniff_time if packets else 'N/A'}")

    # Process captured packets
    for packet in packets:
        try:
            if not hasattr(packet, 'ip'):
                continue

            timestamp = packet.sniff_time
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            proto = int(packet.ip.proto)
            packet_size = int(packet.length)
            
            src_port = int(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0
            dst_port = int(packet.tcp.dstport) if hasattr(packet, 'tcp') else 0

            pkt_entropy = compute_entropy(packet.get_raw_packet())
            session_key = (src_ip, dst_ip, proto)

            if session_data[session_key]["time_start"] is None:
                session_data[session_key]["time_start"] = timestamp
            
            session_data[session_key]["time_end"] = timestamp
            session_data[session_key]["entropy_values"].append(pkt_entropy)

            if hasattr(packet, 'tcp') or hasattr(packet, 'udp'):
                session_data[session_key]["num_pkts_out"] += 1 if int(src_ip.split('.')[0]) < int(dst_ip.split('.')[0]) else 0
                session_data[session_key]["num_pkts_in"] += 1 if int(src_ip.split('.')[0]) > int(dst_ip.split('.')[0]) else 0
                session_data[session_key]["total_bytes_out"] += packet_size if int(src_ip.split('.')[0]) < int(dst_ip.split('.')[0]) else 0
                session_data[session_key]["total_bytes_in"] += packet_size if int(src_ip.split('.')[0]) > int(dst_ip.split('.')[0]) else 0

        except AttributeError as e:
            print(f"Skipping packet due to: {e}")
            continue

    features = []
    for (src_ip, dst_ip, proto), data in session_data.items():
        time_start = data["time_start"]
        time_end = data["time_end"]
        duration = (time_end - time_start).total_seconds() if time_end and time_start else 0
        total_entropy = sum(data["entropy_values"])
        avg_ipt = duration / max(1, (data["num_pkts_out"] + data["num_pkts_in"]))

        features.append({
            "avg_ipt": avg_ipt,
            "bytes_in": data["total_bytes_in"],
            "bytes_out": data["total_bytes_out"],
            "dest_port": dst_port,
            "entropy": total_entropy / max(1, len(data["entropy_values"])),
            "num_pkts_out": data["num_pkts_out"],
            "num_pkts_in": data["num_pkts_in"],
            "proto": proto,
            "src_port": src_port,
            "total_entropy": total_entropy,
            "label": 0,
            "duration": duration,
            "Year": time_start.year if time_start else 0,
            "Month": time_start.month if time_start else 0,
            "Day": time_start.day if time_start else 0,
        })

    df = pd.DataFrame(features)
    print(f"\nProcessed {len(df)} unique sessions")
    print(df.head())

    return df