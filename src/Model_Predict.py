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

import Data_cleanup
import HyperParameters as H
import Utils as U
import Dataset as D
import preprocessing_functions
import Model

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

model = Model.Network_Dection_Model(H.OUTPUT_DIM, #'avg_ipt', 'entropy', 'total_entropy', 'duration', 1 output neuron
                                    embed_dims[0], num_embeds[0], #Bin
                                    embed_dims[1], num_embeds[1], #Bout
                                    embed_dims[2], num_embeds[2], #Pin
                                    embed_dims[3], num_embeds[3], #Pout
                                    embed_dims[4], num_embeds[4]) #Proto
try:
    # Try loading the model weights on the same device (GPU or CPU)
    model.load_state_dict(torch.load((U.MODEL_FOLDER / Model_Name).resolve(), weights_only=True))
except:
    # In case there's a device mismatch, load the model weights on the CPU
    model.load_state_dict(torch.load((U.MODEL_FOLDER / Model_Name).resolve(), weights_only=True, map_location=torch.device('cpu')))
model.to(device)
### LOADED TRAINED MODEL ###

# List to store loaded scalers
scalers = []

# Get all .pkl files in the folder
pkl_files = glob.glob(os.path.join(U.SCALERS_FOLDER, "*.pkl"))

# Loop through each file and load the scaler
for file_path in pkl_files:
    try:
        with open(file_path, 'rb') as f:
            scaler = pickle.load(f)
            scalers.append(scaler)
            print(f"Loaded scaler from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def pipeline(data):

    #'avg_ipt', 'entropy', 'total_entropy', 'duration', [0: 4] 'bytes_in', 4 'bytes_out', 5 'num_pkts_in', 6 'num_pkts_out', 7 'proto', 8
    
    try:
        data = torch.from_numpy(np.array(data).astype(np.float32))
    except:
        pass
    #Scale the continuous data
    data = data.numpy()
    for index in range(4):
        data[index] = scalers[index].transform(data[index].reshape(-1, 1)).flatten()
    data = torch.from_numpy(data.astype(np.float32)) #Turn back to tensor

    #Map discrete data
    data[4] = preprocessing_functions.approximate_value(data[4], model_params['original_bin'])
    data[5] = preprocessing_functions.approximate_value(data[5], model_params['original_bout'])
    data[4] = model_params['feature_map_bin'][str(int(data[4].item()))]
    data[5] = model_params['feature_map_bout'][str(int(data[5].item()))]
    data[6] = model_params['feature_map_bout'][str(int(data[6].item()))]
    data[7] = model_params['feature_map_bout'][str(int(data[7].item()))]
    data[8] = model_params['feature_map_proto'][str(int(data[8].item()))]

    #Add batch dimension
    data = data.unsqueeze(1)
    data = data.T #Transpose batch dimension to the front
    data = data.to(device)

    x = data[:, :9]
    with torch.no_grad():
        y_pred = torch.argmax(F.softmax(model(data))) #Get model prediction

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

def test_model(amount=10): #Test the model on the test data
    #Load test data to test model

    payload = {
        "model": "your-model-name",  # Some versions require this, but it can often be ignored
        "messages": [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "What's up man."}
        ],
        "temperature": 1,
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, json=payload)
    response = response.json()

    assistant_response = response["choices"][0]["message"]["content"]
    print(assistant_response)

    test_data = torch.load(U.test_data, weights_only=True)

    print(test_data.shape)

    correct = 0
    for x in range(amount):
        ridx = random.randint(0, len(test_data)-1) #Random index to test
        d = test_data[:, ridx]
        pred = pipeline(d)
        if pred == d[9]:
            correct += 1

    print(f"% Correct: {(correct/amount)*100:.2f}%")
        


if __name__ == "__main__":
    chat_loop()
    test_model()