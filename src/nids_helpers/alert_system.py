import logging
import json
from datetime import datetime
import requests

class AlertSystem:
    def __init__(self, log_file="ids_alerts.log", llm_endpoint=None):
        # Set up logging
        self.logger = logging.getLogger("IDS_Alerts")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        if llm_endpoint:
            # LLM endpoint and chat history storage
            self.llm_endpoint = llm_endpoint
            self.chat_histories = {}

    def generate_alert(self, threat, packet_info):
        """Generates an alert and optionally sends it to LLM for further analysis."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'threat_type': threat['type'],
            'source_ip': packet_info.get('source_ip'),
            'destination_ip': packet_info.get('destination_ip'),
            'confidence': threat.get('confidence', 0.0),
            'details': threat
        }

        # Log the alert
        self.logger.warning(json.dumps(alert))
        if threat['confidence'] > 0.8:
            self.logger.critical(f"High confidence threat detected: {json.dumps(alert)}")
        
        # Construct the alert context
        alert_context = self._create_alert_context(alert)
        
        # Send to LLM for analysis if confidence is above threshold
        if threat['confidence'] > 0.5:
            try:
                llm_analysis = self._get_llm_analysis(packet_info.get('source_ip'), alert_context)
                self.logger.info(f"LLM Analysis for alert: {json.dumps(llm_analysis)}")
            except Exception as e:
                self.logger.error(f"Failed to get LLM analysis: {str(e)}")

    def _create_alert_context(self, alert):
        """Formats alert data into a structured text summary."""
        return (
            f"Security Alert Details:\n"
            f"- Timestamp: {alert['timestamp']}\n"
            f"- Threat Type: {alert['threat_type']}\n"
            f"- Source IP: {alert['source_ip']}\n"
            f"- Destination IP: {alert['destination_ip']}\n"
            f"- Confidence Score: {alert['confidence']}\n"
            f"- Additional Details: {json.dumps(alert['details'])}\n"
        )

    def _get_llm_analysis(self, user_id, alert_context):
        """Sends the alert to the LLM for analysis using a chat-based paradigm."""
        question = (
            "Analyze this security alert and provide:\n"
            "1) Severity assessment\n"
            "2) Potential impact\n"
            "3) Recommended immediate actions"
        )

        # Ensure a chat history exists for the user/session
        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = [
                {"role": "system", "content": "You are a cybersecurity alert analysis assistant."}
            ]

        # Append the user's alert message
        self.chat_histories[user_id].append({"role": "user", "content": alert_context + "\n" + question})

        try:
            response = requests.post(
                self.llm_endpoint,
                json={"question": question, "context": alert_context},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                print(f"Response status: {response.status_code}, Body: {response.text}")
                raise Exception(f"LLM request failed with status code: {response.status_code}")

            analysis = response.json().get("analysis", "No analysis provided.")
            
            # Append LLM response to history
            self.chat_histories[user_id].append({"role": "assistant", "content": analysis})

            return {
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            return {"error": str(e)}
