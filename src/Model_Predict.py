import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import math
import json
import glob
import random
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import src.HyperParameters as H
import src.Utils as U
import src.preprocessing_functions as preprocessing_functions
from src.Model import Network_Dection_Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


device = H.device
Model_Name = H.MODEL_NAME + '.pth'

url = "http://localhost:5001/v1/chat/completions"  # Replace with your KoboldCPP API URL
headers = {"Content-Type": "application/json"}

### LOAD TRAINED MODEL ###
with open(U.model_params, 'r') as j:
    model_params = json.load(j)

model_params['original_bin'] = torch.from_numpy(np.array(model_params['original_bin']).astype(np.float32)).to(torch.long)
model_params['original_bout'] = torch.from_numpy(np.array(model_params['original_bout']).astype(np.float32)).to(torch.long)

output = model_params['output']
embed_dims = model_params['embed_dims']
num_embeds = model_params['num_embeds']

print(embed_dims)
print(num_embeds)

model = Network_Dection_Model(H.OUTPUT_DIM, #'avg_ipt', 'entropy', 'total_entropy', 'duration', 1 output neuron
                                    embed_dims[0], num_embeds[0], #Bin
                                    embed_dims[1], num_embeds[1], #Bout
                                    embed_dims[2], num_embeds[2], #Pin
                                    embed_dims[3], num_embeds[3], #Pout
                                    embed_dims[4], num_embeds[4]) #Proto
try:
    # Try loading the model weights on the same device (GPU or CPU)
    model.load_state_dict(torch.load((U.MODEL_FOLDER / Model_Name).resolve()))
except:
    # In case there's a device mismatch, load the model weights on the CPU
    model.load_state_dict(torch.load((U.MODEL_FOLDER / Model_Name).resolve(), map_location=torch.device('cpu')))
model.to(device)
model.eval()
### LOADED TRAINED MODEL ###

# List to store loaded scalers
scalers = []

# Get all .pkl files in the folder
pkl_files = sorted(glob.glob(os.path.join(U.SCALERS_FOLDER, "*.pkl")))

# Loop through each file and load the scaler
for file_path in pkl_files:
    try:
        with open(file_path, 'rb') as f:
            scaler = pickle.load(f)
            scalers.append(scaler)
            print(f"Loaded scaler from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(scalers)

def check_pipeline(data):
    #'avg_ipt', 'entropy', 'total_entropy', 'duration', [0: 4] 'bytes_in', 4 'bytes_out', 5 'num_pkts_in', 6 'num_pkts_out', 7 'proto', 8
    try:
        data = torch.from_numpy(np.array(data).astype(np.float32))
    except:
        pass
    #print(data)
    data = data.numpy()
    for index in range(4):
        data[index] = scalers[index].transform(data[index].reshape(-1, 1)).flatten()
    data = torch.from_numpy(data.astype(np.float32)) #Turn back to tensor

    #Map discrete data
    data[4] = preprocessing_functions.approximate_value(data[4], model_params['original_bin'])
    data[5] = preprocessing_functions.approximate_value(data[5], model_params['original_bout'])
    data[4] = model_params['feature_map_bin'][str(int(data[4].item()))]
    data[5] = model_params['feature_map_bout'][str(int(data[5].item()))]
    try:
        data[8] = model_params['feature_map_proto'][str(int(data[8].item()))] 
    except KeyError as e:
        # Create a new list to store the transformed values
        data[8] = model_params['feature_map_proto'][str(0)]

    #Add batch dimension
    data = data.unsqueeze(1)
    data = data.T #Transpose batch dimension to the front
    data = data.to(device)

    x = data[:, :-1]
    with torch.no_grad():
        raw_output = model(x)
        probabilities = F.softmax(raw_output, dim=1)
        y_pred = torch.argmax(probabilities, dim=1)

    print(f'Predicted Class: {H.CLASSES[y_pred]}')
    print(f'Actual Class: {H.CLASSES[int(data[0, 9])]}')
    return y_pred

def chat_loop():
    payload = {
        "model": "your-model-name",  # Some versions require this, but it can often be ignored
        "messages": [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "What's up man."}
        ],
        "temperature": 1,
        "max_tokens": 100
    }
    while(True):

        response = chat_with_ai(payload)
        print(response)

def chat_with_ai(payload):
    response = requests.post(url, headers=headers, json=payload)
    response = response.json()

    assistant_response = response["choices"][0]["message"]["content"]
    return assistant_response

def test_model(amount=10000): #Test the model on the test data
    #Load test data to test model

    test_data = torch.load(U.test_data, weights_only=True).T

    #print(test_data.shape)
    preds = []

    correct = 0
    false_pos = 0
    false_neg = 0
    suspect_neg = 0
    false_pos_on_outlier = 0
    correct = 0
    false_pos = 0
    false_neg = 0
    suspect_neg = 0
    false_pos_on_outlier = 0
    normal_correct = 0
    anomalous_correct = 0
    normal_data = 0
    anomalous_data = 0 
    for x in range(amount):
        ridx = random.randint(0, len(test_data)-1) #Random index to test
        d = test_data[:, ridx]
        pred = check_pipeline(d)
        pred = int(pred.item())
        label = int(d[9].item())
        preds.append(pred)
        if pred == label:
            correct += 1
        elif pred == 2 and label == 0:
            false_pos += 1
        elif pred == 0 and label in {1, 2}:
            false_neg += 1
        elif pred == 1 and label == 2:
            suspect_neg += 1
        elif pred == 2 and label == 1:
            false_pos_on_outlier += 1
        if pred == 0 and label == 0:
            normal_correct += 1
        elif pred != 0 and label != 0:
            anomalous_correct += 1
        if label == 0:
            normal_data += 1
        else:
            anomalous_data += 1
    preds = torch.tensor(preds)

    # Calculate percentages
    correct_pct = (correct / amount) * 100
    false_pos_pct = (false_pos / amount) * 100
    false_neg_pct = (false_neg / amount) * 100
    suspect_neg_pct = (suspect_neg / amount) * 100
    false_pos_outlier_pct = (false_pos_on_outlier / amount) * 100
    normal_correct_pct = (normal_correct / normal_data) * 100
    anomalous_correct_pct = (anomalous_correct / anomalous_data) * 100

    # Print metrics
    print("\n---=== Metrics ===---")
    print(f"% Correct: {correct_pct:.2f}%")
    print(f"False positive: {false_pos_pct:.2f}%")
    print(f"False negative: {false_neg_pct:.2f}%")
    print(f"Outlier predict on malicious: {suspect_neg_pct:.2f}%")
    print(f"Malicious predict on outliers: {false_pos_outlier_pct:.2f}%")
    print(f"Correct on normal data flow: {normal_correct_pct}%")
    print(f"Correcnt on anomalous data flow: {anomalous_correct_pct}")
    print("-----------------------")

    # Visualization
    labels = ["Correct", "False Positives", "False Negatives", "Outlier on malicious", "Mal on Outliers", "IDing Normal Data", "IDing Anomalous Data"]
    values = [correct_pct, false_pos_pct, false_neg_pct, suspect_neg_pct, false_pos_outlier_pct, normal_correct_pct, anomalous_correct_pct]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["green", "red", "blue", "orange", "purple", "Yellow", "Pink"])
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics")
    plt.ylim(0, 100)

    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.2f}%", ha="center", fontsize=10)

    plt.show()

if __name__ == "__main__":
    #chat_loop()
    test_model()