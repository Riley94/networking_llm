import torch
import pandas as pd
import pickle
import os
import numpy as np
import joblib

class DetectionEngine:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'detection_model')
        self.anomaly_detector = torch.load(os.path.join(self.model_path, 'autoencoder.pth'))
        self.binary_classifier = joblib.load(os.path.join(self.model_path, 'binary_classifier.pkl'))
        self.attack_classifier = torch.load(os.path.join(self.model_path, 'best_attack_classifier.pth'))
        self.label_encoder = pickle.load(open(os.path.join(self.model_path, 'label_encoder.pkl'), 'rb'))
        self.scaler = pickle.load(open(os.path.join(self.model_path, 'scaler.pkl'), 'rb'))
        self.threshold = 5737109.5  # Threshold for anomaly detection determined during training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.signature_rules = self.load_signature_rules()

    # Placeholder for loading signature-based rules
    # TODO: Implement this method
    def load_signature_rules(self):
        return {
            'syn_flood': {
                'condition': lambda features: (
                    features['tcp_flags'] == '0x002' and  # SYN flag in pyshark
                    features['packet_rate'] > 100
                )
            },
            'port_scan': {
                'condition': lambda features: (
                    features['packet_size'] < 100 and
                    features['packet_rate'] > 50
                )
            }
        }

    def detect_threats(self, features):
        threats = []

        feature_df = pd.DataFrame([features])
        protos = ['proto_6', 'proto_17', 'proto_47', 'proto_58', 'proto_50', 'proto_51', 'proto_132', 'proto_89']
        feature_df[protos] = 0
        
        # Handle protocol field using one-hot encoding
        if 'proto' in feature_df.columns:
            proto_num = feature_df['proto']
            proto_col = 'proto_' + str(proto_num)
            feature_df[proto_col] = 1
            feature_df.drop(['proto'], axis=1, inplace=True)
        
        # Drop fields not used in training (similar to preprocessing pipeline)
        if 'src_ip' in feature_df.columns:
            feature_df.drop(['src_ip'], axis=1, inplace=True)
        if 'dest_ip' in feature_df.columns:
            feature_df.drop(['dest_ip'], axis=1, inplace=True)
        if 'time_start' in feature_df.columns:
            feature_df.drop(['time_start'], axis=1, inplace=True)
        if 'time_end' in feature_df.columns:
            feature_df.drop(['time_end'], axis=1, inplace=True)
        
        # Drop date fields not used by the model
        feature_df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
        
        # Ensure ports are properly formatted
        feature_df.dest_port = feature_df.dest_port.fillna(-1)
        feature_df.dest_port = feature_df.dest_port.infer_objects(copy=False).astype('int64')
        feature_df.src_port = feature_df.src_port.fillna(-1)
        feature_df.src_port = feature_df.src_port.infer_objects(copy=False).astype('int64')
        
        # Scale the numeric features using the same scaler used during training
        cols_to_scale = ['avg_ipt', 'bytes_in', 'bytes_out', 'num_pkts_in', 'num_pkts_out', 
                        'total_entropy', 'entropy', 'duration']
        
        # Apply scaling only to columns that exist in the input
        scale_cols = [col for col in cols_to_scale if col in feature_df.columns]
        if scale_cols:
            feature_df[scale_cols] = self.scaler.transform(feature_df[scale_cols])
        
        # Convert to PyTorch tensor
        feature_tensor = torch.tensor(feature_df.to_numpy(), dtype=torch.float32)
        
        # Set model to evaluation mode and run inference
        self.anomaly_detector.eval()
        with torch.no_grad():
            # Move tensor to the appropriate device (CPU or GPU)
            feature_tensor = feature_tensor.to(self.device)
            
            # Get reconstruction from autoencoder
            reconstructed = self.anomaly_detector(feature_tensor)
            
            # Calculate reconstruction error (MSE)
            reconstruction_error = torch.mean((reconstructed - feature_tensor) ** 2, dim=1).item()
            
            # Check if error exceeds threshold
            if reconstruction_error >= self.threshold:
                threats.append({
                    'type': 'anomaly',
                    'score': reconstruction_error,
                    'confidence': min(1.0, reconstruction_error / (self.threshold * 2))  # Normalize confidence
                })
                label = self.binary_classifier.predict(feature_tensor)
        
        return threats

    def extract_features_from_pyshark(self, packets):
        """
        Convert pyshark packets to a dataframe matching the format of the training data
        
        Args:
            packets: List of pyshark packet objects
            
        Returns:
            DataFrame with features extracted from packets
        """
        flows = {}  # Dictionary to store flow information
        
        for packet in packets:
            try:
                # Only process IPv4 packets with TCP, UDP, or ICMP
                if not hasattr(packet, 'ip'):
                    continue
                    
                # Basic flow identification (5-tuple)
                if hasattr(packet, 'tcp'):
                    protocol = 6  # TCP
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    tcp_flags = int(packet.tcp.flags, 16)
                    tcp_window = int(packet.tcp.window_size)
                elif hasattr(packet, 'udp'):
                    protocol = 17  # UDP
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                    tcp_flags = 0
                    tcp_window = 0
                elif hasattr(packet, 'icmp'):
                    protocol = 1  # ICMP
                    src_port = 0
                    dst_port = 0
                    tcp_flags = 0
                    tcp_window = 0
                    icmp_type = int(packet.icmp.type) if hasattr(packet.icmp, 'type') else 0
                else:
                    continue
                    
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                
                # Create bidirectional flow key
                if (src_ip, src_port, dst_ip, dst_port, protocol) < (dst_ip, dst_port, src_ip, src_port, protocol):
                    flow_key = (src_ip, src_port, dst_ip, dst_port, protocol)
                    direction = 'forward'
                else:
                    flow_key = (dst_ip, dst_port, src_ip, src_port, protocol)
                    direction = 'backward'
                    
                # Get packet length and TTL
                pkt_len = int(packet.length)
                ttl = int(packet.ip.ttl)
                
                # Initialize flow if new
                if flow_key not in flows:
                    flows[flow_key] = {
                        'IPV4_SRC_ADDR': flow_key[0],
                        'L4_SRC_PORT': flow_key[1],
                        'IPV4_DST_ADDR': flow_key[2],
                        'L4_DST_PORT': flow_key[3],
                        'PROTOCOL': flow_key[4],
                        'L7_PROTO': 0.0,  # Default value
                        'IN_BYTES': 0,
                        'IN_PKTS': 0,
                        'OUT_BYTES': 0,
                        'OUT_PKTS': 0,
                        'TCP_FLAGS': 0,
                        'CLIENT_TCP_FLAGS': 0,
                        'SERVER_TCP_FLAGS': 0,
                        'FLOW_DURATION_MILLISECONDS': 0,
                        'DURATION_IN': 0,
                        'DURATION_OUT': 0,
                        'MIN_TTL': ttl,
                        'MAX_TTL': ttl,
                        'LONGEST_FLOW_PKT': pkt_len,
                        'SHORTEST_FLOW_PKT': pkt_len,
                        'MIN_IP_PKT_LEN': pkt_len,
                        'MAX_IP_PKT_LEN': pkt_len,
                        'SRC_TO_DST_SECOND_BYTES': 0.0,
                        'DST_TO_SRC_SECOND_BYTES': 0.0,
                        'RETRANSMITTED_IN_BYTES': 0,
                        'RETRANSMITTED_IN_PKTS': 0,
                        'RETRANSMITTED_OUT_BYTES': 0,
                        'RETRANSMITTED_OUT_PKTS': 0,
                        'SRC_TO_DST_AVG_THROUGHPUT': 0,
                        'DST_TO_SRC_AVG_THROUGHPUT': 0,
                        'NUM_PKTS_UP_TO_128_BYTES': 0,
                        'NUM_PKTS_128_TO_256_BYTES': 0,
                        'NUM_PKTS_256_TO_512_BYTES': 0,
                        'NUM_PKTS_512_TO_1024_BYTES': 0,
                        'NUM_PKTS_1024_TO_1514_BYTES': 0,
                        'TCP_WIN_MAX_IN': 0,
                        'TCP_WIN_MAX_OUT': 0,
                        'ICMP_TYPE': 0,
                        'ICMP_IPV4_TYPE': 0,
                        'DNS_QUERY_ID': 0,
                        'DNS_QUERY_TYPE': 0,
                        'DNS_TTL_ANSWER': 0,
                        'FTP_COMMAND_RET_CODE': 0.0,
                        'start_time': float(packet.sniff_timestamp),
                        'last_time': float(packet.sniff_timestamp),
                    }
                else:
                    # Update flow duration
                    flows[flow_key]['FLOW_DURATION_MILLISECONDS'] = int((float(packet.sniff_timestamp) - flows[flow_key]['start_time']) * 1000)
                    flows[flow_key]['last_time'] = max(flows[flow_key]['last_time'], float(packet.sniff_timestamp))
                    
                    # Update min/max values
                    flows[flow_key]['MIN_TTL'] = min(flows[flow_key]['MIN_TTL'], ttl)
                    flows[flow_key]['MAX_TTL'] = max(flows[flow_key]['MAX_TTL'], ttl)
                    flows[flow_key]['MIN_IP_PKT_LEN'] = min(flows[flow_key]['MIN_IP_PKT_LEN'], pkt_len)
                    flows[flow_key]['MAX_IP_PKT_LEN'] = max(flows[flow_key]['MAX_IP_PKT_LEN'], pkt_len)
                    flows[flow_key]['SHORTEST_FLOW_PKT'] = min(flows[flow_key]['SHORTEST_FLOW_PKT'], pkt_len)
                    flows[flow_key]['LONGEST_FLOW_PKT'] = max(flows[flow_key]['LONGEST_FLOW_PKT'], pkt_len)
                
                # Update packet counts and bytes based on direction
                if direction == 'forward':
                    flows[flow_key]['IN_BYTES'] += pkt_len
                    flows[flow_key]['IN_PKTS'] += 1
                    flows[flow_key]['DURATION_IN'] = int((float(packet.sniff_timestamp) - flows[flow_key]['start_time']) * 1000)
                    
                    if hasattr(packet, 'tcp'):
                        flows[flow_key]['CLIENT_TCP_FLAGS'] |= tcp_flags
                        flows[flow_key]['TCP_WIN_MAX_IN'] = max(flows[flow_key]['TCP_WIN_MAX_IN'], tcp_window)
                else:
                    flows[flow_key]['OUT_BYTES'] += pkt_len
                    flows[flow_key]['OUT_PKTS'] += 1
                    flows[flow_key]['DURATION_OUT'] = int((float(packet.sniff_timestamp) - flows[flow_key]['start_time']) * 1000)
                    
                    if hasattr(packet, 'tcp'):
                        flows[flow_key]['SERVER_TCP_FLAGS'] |= tcp_flags
                        flows[flow_key]['TCP_WIN_MAX_OUT'] = max(flows[flow_key]['TCP_WIN_MAX_OUT'], tcp_window)
                
                # Update TCP flags
                if hasattr(packet, 'tcp'):
                    flows[flow_key]['TCP_FLAGS'] |= tcp_flags
                    
                # Update ICMP type
                if hasattr(packet, 'icmp'):
                    flows[flow_key]['ICMP_TYPE'] = icmp_type
                    flows[flow_key]['ICMP_IPV4_TYPE'] = icmp_type
                    
                # Update packet size distribution
                if pkt_len <= 128:
                    flows[flow_key]['NUM_PKTS_UP_TO_128_BYTES'] += 1
                elif pkt_len <= 256:
                    flows[flow_key]['NUM_PKTS_128_TO_256_BYTES'] += 1
                elif pkt_len <= 512:
                    flows[flow_key]['NUM_PKTS_256_TO_512_BYTES'] += 1
                elif pkt_len <= 1024:
                    flows[flow_key]['NUM_PKTS_512_TO_1024_BYTES'] += 1
                else:
                    flows[flow_key]['NUM_PKTS_1024_TO_1514_BYTES'] += 1
                    
                # Check for DNS
                if hasattr(packet, 'dns'):
                    flows[flow_key]['L7_PROTO'] = 53.0  # DNS protocol
                    if hasattr(packet.dns, 'id'):
                        flows[flow_key]['DNS_QUERY_ID'] = int(packet.dns.id)
                    if hasattr(packet.dns, 'qry_type'):
                        flows[flow_key]['DNS_QUERY_TYPE'] = int(packet.dns.qry_type)
                    if hasattr(packet.dns, 'resp_ttl'):
                        flows[flow_key]['DNS_TTL_ANSWER'] = int(packet.dns.resp_ttl)
                        
                # Check for FTP
                if hasattr(packet, 'ftp') and hasattr(packet.ftp, 'response_code'):
                    flows[flow_key]['L7_PROTO'] = 21.0  # FTP protocol
                    flows[flow_key]['FTP_COMMAND_RET_CODE'] = float(packet.ftp.response_code)
                    
            except Exception as e:
                print(f"Error processing packet: {e}")
                continue
        
        # Calculate derived fields for each flow
        for flow_key, flow in flows.items():
            # Calculate throughput
            duration_sec = max(flow['FLOW_DURATION_MILLISECONDS'] / 1000, 0.001)  # Avoid division by zero
            flow['SRC_TO_DST_AVG_THROUGHPUT'] = int(flow['IN_BYTES'] / duration_sec)
            flow['DST_TO_SRC_AVG_THROUGHPUT'] = int(flow['OUT_BYTES'] / duration_sec)
            
            # Calculate bytes per second
            flow['SRC_TO_DST_SECOND_BYTES'] = flow['IN_BYTES'] / duration_sec
            flow['DST_TO_SRC_SECOND_BYTES'] = flow['OUT_BYTES'] / duration_sec
            
            # Remove temporary fields
            flow.pop('start_time', None)
            flow.pop('last_time', None)
            
        # Convert to DataFrame
        df = pd.DataFrame(list(flows.values()))
        
        # Add empty columns for any missing fields
        required_columns = [
            'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL',
            'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS',
            'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
            'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT',
            'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
            'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
            'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
            'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
            'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
            'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
            'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
            'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
            'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                if col in ['L7_PROTO', 'FTP_COMMAND_RET_CODE']:
                    df[col] = 0.0
                else:
                    df[col] = 0
        
        # Use same dtypes as training data
        dtype_map = {
            'L4_SRC_PORT': 'int32',
            'L4_DST_PORT': 'int32',
            'PROTOCOL': 'int16',
            'L7_PROTO': 'float16',
            'IN_BYTES': 'int32',
            'IN_PKTS': 'int32',
            'OUT_BYTES': 'int32',
            'OUT_PKTS': 'int32',
            'TCP_FLAGS': 'int16',
            'CLIENT_TCP_FLAGS': 'int16',
            'SERVER_TCP_FLAGS': 'int16',
            'FLOW_DURATION_MILLISECONDS': 'int32',
            'DURATION_IN': 'int32',
            'DURATION_OUT': 'int32',
            'MIN_TTL': 'int16',
            'MAX_TTL': 'int16',
            'LONGEST_FLOW_PKT': 'int16',
            'SHORTEST_FLOW_PKT': 'int16',
            'MIN_IP_PKT_LEN': 'int16',
            'MAX_IP_PKT_LEN': 'int16',
            'SRC_TO_DST_SECOND_BYTES': 'float64',
            'DST_TO_SRC_SECOND_BYTES': 'float64',
            'RETRANSMITTED_IN_BYTES': 'int32',
            'RETRANSMITTED_IN_PKTS': 'int16',
            'RETRANSMITTED_OUT_BYTES': 'int32',
            'RETRANSMITTED_OUT_PKTS': 'int16',
            'SRC_TO_DST_AVG_THROUGHPUT': 'int32',
            'DST_TO_SRC_AVG_THROUGHPUT': 'int64',
            'NUM_PKTS_UP_TO_128_BYTES': 'int32',
            'NUM_PKTS_128_TO_256_BYTES': 'int16',
            'NUM_PKTS_256_TO_512_BYTES': 'int16',
            'NUM_PKTS_512_TO_1024_BYTES': 'int32',
            'NUM_PKTS_1024_TO_1514_BYTES': 'int32',
            'TCP_WIN_MAX_IN': 'int32',
            'TCP_WIN_MAX_OUT': 'int32',
            'ICMP_TYPE': 'int32',
            'ICMP_IPV4_TYPE': 'int16',
            'DNS_QUERY_ID': 'int32',
            'DNS_QUERY_TYPE': 'int16',
            'DNS_TTL_ANSWER': 'int64',
            'FTP_COMMAND_RET_CODE': 'float16'
        }
        
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
                
        # Add additional feature columns
        df['bytes_ratio'] = df['IN_BYTES'] / (df['OUT_BYTES'] + 1)
        df['pkts_ratio'] = df['IN_PKTS'] / (df['OUT_PKTS'] + 1)
        df['bytes_per_pkt_in'] = df['IN_BYTES'] / (df['IN_PKTS'] + 1)
        df['bytes_per_pkt_out'] = df['OUT_BYTES'] / (df['OUT_PKTS'] + 1)
        
        # Add placeholder columns for compatibility (these will be filled by the model during prediction)
        df['Label'] = 0  # Will be predicted
        df['Attack'] = 'Normal'  # Will be predicted
        df['Dataset'] = 'Live'
        df['Attack_encoded'] = 0  # Will be updated after prediction
        
        return df

    def preprocess_pyshark_data(self, packets, saved_preprocessor, attack_encoder):
        """
        Process pyshark packets to match the format of the training data
        and apply the same preprocessing steps
        
        Args:
            packets: List of pyshark packet objects
            saved_preprocessor: The ColumnTransformer from the training process
            attack_encoder: The LabelEncoder used for encoding attack types
            
        Returns:
            X: Preprocessed features ready for model prediction
            df: Original dataframe with all features
        """
        # Convert pyshark packets to dataframe with matching format
        df = self.extract_features_from_pyshark(packets)
        
        # Get the same columns as used in training
        categorical_cols = ['PROTOCOL']
        ip_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
        exclude_cols = ip_cols + ['Label', 'Attack', 'Dataset', 'Attack_encoded']
        
        # Create the derived features
        df['bytes_ratio'] = df['IN_BYTES'] / (df['OUT_BYTES'] + 1)
        df['pkts_ratio'] = df['IN_PKTS'] / (df['OUT_PKTS'] + 1)
        df['bytes_per_pkt_in'] = df['IN_BYTES'] / (df['IN_PKTS'] + 1)
        df['bytes_per_pkt_out'] = df['OUT_BYTES'] / (df['OUT_PKTS'] + 1)
        
        # Define numerical columns
        numerical_cols = [col for col in df.columns if col not in categorical_cols + exclude_cols]
        
        # Create X by selecting only the feature columns
        X = df[numerical_cols + categorical_cols]
        
        # Apply the saved preprocessor to transform the data
        X_transformed = saved_preprocessor.transform(X)
        
        return X_transformed, df

    def predict_with_model(self, X_transformed, df):
        """
        Use the trained model to make predictions on the preprocessed data
        
        Args:
            model: Trained machine learning model
            X_transformed: Preprocessed features
            df: Original dataframe with all features
            attack_encoder: LabelEncoder used for attack types
            
        Returns:
            DataFrame with predictions
        """
        # Make binary predictions (normal vs attack)
        binary_predictions = self.binary_classifier.predict(X_transformed)
        
        # Update the dataframe with predictions
        df['Label'] = binary_predictions
        
        # If any attacks are detected, predict their type
        if any(binary_predictions == 1):
            # Make attack type predictions
            attack_predictions = self.attack_classifier.predict(X_transformed[binary_predictions == 1])
            
            # Decode the attack types
            decoded_attacks = self.label_encoder.inverse_transform(attack_predictions)
            
            # Update the dataframe with attack types
            df.loc[binary_predictions == 1, 'Attack'] = decoded_attacks
            df.loc[binary_predictions == 1, 'Attack_encoded'] = attack_predictions
        
        return df

    # Example usage:
    """
    import pyshark
    import pickle

    # Load your saved model and preprocessor
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
        
    with open('attack_encoder.pkl', 'rb') as f:
        attack_encoder = pickle.load(f)

    # Capture packets with pyshark
    capture = pyshark.LiveCapture(interface='eth0')
    capture.sniff(timeout=60)  # Capture for 60 seconds

    # Process the captured packets
    X_transformed, df = preprocess_pyshark_data(capture, preprocessor, attack_encoder)

    # Make predictions
    results = predict_with_model(model, X_transformed, df, attack_encoder)

    # Show detected attacks
    attacks = results[results['Label'] == 1]
    print(f"Detected {len(attacks)} potential attacks:")
    print(attacks[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']])
    """