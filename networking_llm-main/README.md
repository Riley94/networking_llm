# Network Intrusion Detection System (NIDS)

## Overview
This project is a **Network Intrusion Detection System (NIDS)** designed to monitor network traffic, detect anomalies, and identify potential threats. It utilizes **PyShark** for packet capture and **Isolation Forest** for anomaly detection. Future enhancements include integrating a **Large Language Model (LLM) with Retrieval-Augmented Generation (RAG)** to provide more intelligent threat analysis.

## Features
- **Traffic Capture**: Uses PyShark to capture and analyze network packets in real time.
- **Anomaly Detection**: Implements an **Isolation Forest** model to detect deviations from normal traffic patterns.
- **Signature-Based Detection**: Rules-based approach for identifying common attack patterns like SYN floods and port scans.
- **Extensible Design**: Plans to integrate an **LLM with RAG** for enhanced threat detection and explanation.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8). Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/Riley94/networking_llm.git
cd nids/src

# Install dependencies
pip install -r requirements.txt
```

## Usage
To start the IDS, run the following command:

```bash
python src/nids.py
```

This will initiate network packet capture and begin monitoring for threats.

## Project Structure
```
networking_llm/
├── src/
|   ├── nids.py  # Main entrypoint
|   ├── nids_helpers/
│   |   ├── packet_capture.py  # Handles traffic capture
│   |   ├── traffic_analyzer.py  # Extracts features from packets
│   |   ├── detection_engine.py  # Implements signature and anomaly detection
│   |   ├── alert_system.py  # Handles alerts for detected threats
│   └── requirements.txt  # Required dependencies
└── README.md  # Project documentation
```

## Future Enhancements
- **Integrate LLM with RAG** to provide contextual threat explanations.
- **Improve visualization** of detected threats.
- **Enhance anomaly detection models** with additional feature engineering.

## License
This project is licensed under the MIT License.

---
*Authors: Riley Bruce, Katy Bohanan, Ransom Ward, Prathyusha Adari*  

