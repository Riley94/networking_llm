import torch
import pandas as pd
import pickle
import os

class DetectionEngine:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'detection_model')
        self.anomaly_detector = torch.load(os.path.join(self.model_path, 'autoencoder.pth'))
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
        
        return threats