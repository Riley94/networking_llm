from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np

class TrafficAnalyzer:
    def __init__(self, flow_timeout: int = 60):
        self.flow_timeout = flow_timeout
        self.flow_stats = defaultdict(lambda: {
            'num_pkts_out': 0,
            'num_pkts_in': 0,
            'bytes_in': 0,
            'bytes_out': 0,
            'start_time': None,
            'last_time': None,
            'packet_times': [],  # For IPT calculation
            'payloads': [],  # For entropy calculation
            'is_active': True,
            'fin_seen': False,
            'rst_seen': False
        })
        self.completed_flows: List[Dict] = []

    def compute_entropy(self, payloads: List[bytes]) -> float:
        if not payloads:
            return 0.0
        
        # Concatenate all payloads
        all_bytes = b''.join(payloads)
        if not all_bytes:
            return 0.0
            
        # Count frequency of each byte
        byte_counts = defaultdict(int)
        for byte in all_bytes:
            byte_counts[byte] += 1
            
        total_bytes = len(all_bytes)
        
        # Calculate Shannon entropy
        entropy = 0
        for count in byte_counts.values():
            p = count / total_bytes
            entropy -= p * np.log2(p)
            
        return entropy

    def calculate_avg_ipt(self, packet_times: List[float]) -> float:
        if len(packet_times) < 2:
            return 0.0
        
        # Calculate inter-packet times
        ipt_values = [j-i for i, j in zip(packet_times[:-1], packet_times[1:])]
        return np.mean(ipt_values)

    def check_flow_termination(self, flow_key: Tuple, current_time: float) -> bool:
        stats = self.flow_stats[flow_key]
        
        if (current_time - stats['last_time']) > self.flow_timeout:
            return True
            
        if stats['fin_seen'] or stats['rst_seen']:
            return True
            
        return False

    def finalize_flow(self, flow_key: Tuple, proto: int, src_port: int, dest_port: int) -> Dict:
        stats = self.flow_stats[flow_key]
        
        current_time = datetime.now()
        
        # Calculate flow entropy
        flow_entropy = self.compute_entropy(stats['payloads'])
        total_entropy = flow_entropy * (stats['num_pkts_in'] + stats['num_pkts_out'])
        
        # Calculate duration and average IPT
        duration = stats['last_time'] - stats['start_time']
        duration = max(duration, 1e-6)
        avg_ipt = self.calculate_avg_ipt(stats['packet_times'])
        
        flow_features = {
            "avg_ipt": avg_ipt,
            "bytes_in": stats['bytes_in'],
            "bytes_out": stats['bytes_out'],
            "dest_port": dest_port,
            "entropy": flow_entropy,
            "num_pkts_out": stats['num_pkts_out'],
            "num_pkts_in": stats['num_pkts_in'],
            "proto": proto,
            "src_port": src_port,
            "total_entropy": total_entropy,
            "label": 0,  # Default label
            "duration": duration,
            "Year": current_time.year,
            "Month": current_time.month,
            "Day": current_time.day
        }
        
        # Add to completed flows and cleanup
        self.completed_flows.append(flow_features)
        del self.flow_stats[flow_key]
        
        return flow_features

    def analyze_packet(self, packet) -> Dict:
        if hasattr(packet, 'ip') and hasattr(packet, 'tcp'):
            ip_src = packet.ip.src
            ip_dst = packet.ip.dst
            port_src = int(packet.tcp.srcport)
            port_dst = int(packet.tcp.dstport)
            proto = int(packet.ip.proto)
            packet_size = int(packet.length)
            flow_key = (ip_src, ip_dst)  # Using IP pairs as flow key
            
            current_time = float(packet.sniff_timestamp)
            
            # Update flow statistics
            stats = self.flow_stats[flow_key]
            
            if not stats['start_time']:
                stats['start_time'] = current_time
            
            stats['last_time'] = current_time
            stats['packet_times'].append(current_time)
            
            # Determine direction and update counters
            if ip_src < ip_dst:  # Outbound
                stats['num_pkts_out'] += 1
                stats['bytes_out'] += packet_size
            else:  # Inbound
                stats['num_pkts_in'] += 1
                stats['bytes_in'] += packet_size
            
            # Store payload for entropy calculation
            if hasattr(packet.tcp, 'payload'):
                try:
                    payload = packet.get_raw_packet()
                    stats['payloads'].append(payload)
                except (ValueError, AttributeError):
                    pass

            # Check TCP flags
            if hasattr(packet.tcp, 'flags'):
                try:
                    flags = int(packet.tcp.flags, 16)  # Convert to an integer
                    if flags & 0x01:  # FIN flag
                        stats['fin_seen'] = True
                    if flags & 0x04:  # RST flag
                        stats['rst_seen'] = True
                except ValueError:
                    print("Could not parse TCP flags:", packet.tcp.flags)

            # Check for flow termination
            if self.check_flow_termination(flow_key, current_time):
                return self.finalize_flow(flow_key, proto, port_src, port_dst)
            
            # For active flows, return current features
            current_entropy = self.compute_entropy(stats['payloads'])
            current_duration = current_time - stats['start_time']
            current_time_obj = datetime.now()
            avg_ipt = self.calculate_avg_ipt(stats['packet_times'])
            print("Returning features")
            return {
                "avg_ipt": avg_ipt,
                "bytes_in": stats['bytes_in'],
                "bytes_out": stats['bytes_out'],
                "dest_port": port_dst,
                "entropy": current_entropy,
                "num_pkts_out": stats['num_pkts_out'],
                "num_pkts_in": stats['num_pkts_in'],
                "proto": proto,
                "src_port": port_src,
                "total_entropy": current_entropy * (stats['num_pkts_in'] + stats['num_pkts_out']),
                "duration": current_duration,
                "Year": current_time_obj.year,
                "Month": current_time_obj.month,
                "Day": current_time_obj.day
            }

        return None

    def get_completed_flows(self) -> List[Dict]:
        return self.completed_flows
