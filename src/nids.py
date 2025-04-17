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
            llm_endpoint="https://cbf6-2605-a601-af2e-da00-124d-6617-5893-a3ef.ngrok-free.app"
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
                packet_features = self.traffic_analyzer.analyze_packet(packet)
                # flow queue
                flow_features = self.traffic_analyzer.get_completed_flows()
                
                if flow_features:
                    # Detect threats based on features
                    threat = self.detection_engine.detect_threats(flow_features.get())
                    print(f"Detected threat: {threat}")
                    if threat:
                        packet_info = {
                            'source_ip': packet.ip.src,
                            'destination_ip': packet.ip.dst,
                            'source_port': int(packet.tcp.srcport),
                            'destination_port': int(packet.tcp.dstport),
                            'protocol': int(packet.ip.proto),
                            'timestamp': packet.sniff_time
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
    
    # Start the IDS
    ids.start()