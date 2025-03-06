from nids_helpers.packet_capture import PacketCapture
from nids_helpers.traffic_analyzer import TrafficAnalyzer
from nids_helpers.detection_engine import DetectionEngine
from nids_helpers.alert_system import AlertSystem
import queue
import subprocess
from data_loading.load_luflow import get_luflow

class IntrusionDetectionSystem:
    def __init__(self):
        self.packet_capture = PacketCapture()
        self.traffic_analyzer = TrafficAnalyzer()
        self.detection_engine = DetectionEngine()
        self.alert_system = AlertSystem(
            log_file="ids_alerts.log",
            #llm_endpoint="[YOUR_GROK_ENDPOINT]/analyze_alert"
        )

    def start(self):
        interface = choose_interface()
        if interface:
            print(f"Starting IDS on interface {interface}")
            self.packet_capture.start_capture(interface)
        else:
            print("No valid interface selected. Exiting.")
            return
        
        while True:
            try:
                # Get packet with timeout
                packet = self.packet_capture.packet_queue.get(timeout=1)
                # Analyze packet and extract features
                features = self.traffic_analyzer.analyze_packet(packet)
                
                if features:
                    # Detect threats based on features
                    threats = self.detection_engine.detect_threats(features)
                    print(f"Detected threats: {threats}")
                    for threat in threats:
                        # Create packet info dictionary using pyshark attributes
                        packet_info = {
                            'source_ip': packet.ip.src,
                            'destination_ip': packet.ip.dst,
                            'source_port': int(packet.tcp.srcport),
                            'destination_port': int(packet.tcp.dstport)
                        }
                        
                        # Generate alert for detected threat
                        self.alert_system.generate_alert(threat, packet_info)
                        
            except queue.Empty:
                continue
            except AttributeError as e:
                # Handle packets that don't have expected attributes
                print(f"Skipping packet due to missing attributes: {e}")
                continue
            except KeyboardInterrupt:
                print("\nStopping IDS...")
                self.packet_capture.stop()
                break
            except Exception as e:
                print(f"Unexpected error processing packet: {e}")
                continue

def get_interfaces():
    """Retrieve available network interfaces using tshark."""
    try:
        output = subprocess.check_output(["tshark", "-D"]).decode("utf-8")
        interfaces = [line.split(". ", 1)[1].split(" ")[0] for line in output.strip().split("\n")]
        return interfaces
    except subprocess.CalledProcessError:
        print("Error: Unable to retrieve network interfaces. Ensure TShark is installed and has the necessary permissions.")
        return []
    
def choose_interface():
    interfaces = get_interfaces()

    if not interfaces:
        print("No interfaces found. Exiting.")
        return
    
    print("Available network interfaces:")
    for idx, iface in enumerate(interfaces, start=1):
        print(f"{idx}. {iface}")

    while True:
        try:
            choice = int(input("Select an interface (number): "))
            if 1 <= choice <= len(interfaces):
                selected_interface = interfaces[choice - 1]
                break
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return selected_interface

if __name__ == "__main__":
    # Initialize IDS with default interface
    ids = IntrusionDetectionSystem()
    
    # Train the anomaly detector if needed
    TRAIN = True
    if TRAIN:
        print("Loading training data...")
        train_data = get_luflow(num_rows=75_000)
        print("Training anomaly detector...")
        ids.detection_engine.train_anomaly_detector(train_data)
    
    # Start the IDS
    ids.start()