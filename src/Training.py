import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import math
import json

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
print(device)

cols = ['avg_ipt', 'entropy', 'total_entropy', 'duration','bytes_in', 'bytes_out', 'num_pkts_in', 'num_pkts_out', 'proto', 'label']

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    data = torch.load(U.clean_data, weights_only=True)
    pass

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    data = torch.load(U.clean_data, weights_only=True)

def reduce_dataset(data, max_samples_per_class=None):
    labels = data[:, -1]
    unique_labels, label_counts = torch.unique(labels, return_counts=True)

    if max_samples_per_class is None:
        max_samples_per_class = label_counts.min().item()

    print("Initial Label Distribution:")
    for label, count in zip(unique_labels, label_counts):
        print(f"Label {label.item()}: {count.item()} samples")
    
    balanced_mask = torch.zeros(len(data), dtype=torch.bool)
    
    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        
        num_samples_to_keep = min(max_samples_per_class, len(label_indices))  # Fix out-of-bounds
        selected_indices = torch.randperm(len(label_indices))[:num_samples_to_keep]
        
        balanced_mask[label_indices[selected_indices]] = True

    reduced_data = data[balanced_mask]

    final_labels = reduced_data[:, -1]
    final_unique, final_counts = torch.unique(final_labels, return_counts=True)

    print("Reduced Dataset Label Distribution:")
    for label, count in zip(final_unique, final_counts):
        print(f"Label {label.item()}: {count.item()} samples")

    return reduced_data

def separate_by_label(data):
    # Separate features and labels
    X = data[:, :-1]  # All columns except the last one (features)
    y = data[:, -1]   # Last column (labels)

    # Get unique labels
    unique_labels = torch.unique(y)

    # Dictionary to store tensors for each label
    label_tensors = {label.item(): X[y == label] for label in unique_labels}

    return label_tensors

###Finish loading data###

def save_scalers(scalers, save_dir='./scalers/'):
    """
    Save all scalers to pickle files.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save scalers
    for i, scaler in enumerate(scalers):
        with open(f'{save_dir}{i}X1_scaler_{H.continuous_variables[i]}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

print(data[0])

for x in range(len(data)):
    check_feature = x
    print(f'min col {cols[x]}: {data[check_feature, torch.argmin(data[check_feature])]}, max: {data[check_feature, torch.argmax(data[check_feature])]}')

data = data.T
data = reduce_dataset(data, 150000)
# If labels are integers
unique_labels, counts = torch.unique(data[:, -1], return_counts=True)

# Print the counts
for label, count in zip(unique_labels, counts):
    print(f"Label {label.item()}: {count.item()} occurrences")

data = data.T

print(f'data_shape: {data.shape}')

print(data[:, -1])

### Make train test split ###
# Transpose to make it (samples, features) for sklearn

data_transposed = data.T
print(f"data_transposed {data_transposed.shape}")

# Split with random state for reproducibility
print("Making train test split")
train_transposed, test_transposed = train_test_split(
    data_transposed, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

print(train_transposed.shape)
print(test_transposed.shape)

# Transpose back to original orientation (features, samples)
train = train_transposed
test = test_transposed

#Save test tensor for testing the model inference
torch.save(test, U.test_data)

#Turn into numpy arrays for sklearn
train = train.numpy().T
test = test.numpy().T

print(train.shape)
print(test.shape)

### Scale Continuous Variables ###
# 
train_scalers = [] #Store scalars to be saved and used on new data

for index in range(4):
    print(f'index: {index}')
    print(train[index].shape)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Create a new scaler
    train[index, :] = scaler.fit_transform(train[index].reshape(-1, 1)).flatten()
    test[index, :] = scaler.transform(test[index].reshape(-1, 1)).flatten()
    train_scalers.append(scaler)

print(train.shape)
print(test.shape)

save_scalers(train_scalers)

#Turn back into tensors
train = torch.from_numpy(train.astype(np.float32))
test = torch.from_numpy(test.astype(np.float32))

for x in range(len(data)):
    check_feature = x
    print(f'min col {cols[x]}: {train[check_feature, torch.argmin(train[check_feature])]}, max: {train[check_feature, torch.argmax(train[check_feature])]}')
    print(f'min col {cols[x]}: {test[check_feature, torch.argmin(test[check_feature])]}, max: {test[check_feature, torch.argmax(test[check_feature])]}')

# If labels are integers
unique_labels, counts = torch.unique(train[-1, :], return_counts=True)

# Print the counts
for label, count in zip(unique_labels, counts):
    print(f"Train Labels {label.item()}: {count.item()} occurrences")

# If labels are integers
unique_labels, counts = torch.unique(test[-1, :], return_counts=True)

# Print the counts
for label, count in zip(unique_labels, counts):
    print(f"Test Labels {label.item()}: {count.item()} occurrences")

'''train = train.T
test = test.T'''

print(train.shape)
print(test.shape)

"""
'avg_ipt', 'entropy', 'total_entropy', 'duration', [0: 4]
'bytes_in', 4
'bytes_out', 5
'num_pkts_in', 6
'num_pkts_out', 7
'proto', 8
"""

### Make Dataset ###
Train_Dataset = D.LUFlow_ND_Dataset(train)
Test_Dataset = D.LUFlow_ND_Dataset(test)

### Map Values ###
print("Mapping values")
feature_map_bin, feature_map_bout, feature_map_proto, original_bin, original_bout = preprocessing_functions.map_values(Train_Dataset)

# Explicitly assign back to ensure updates stick
Train_Dataset.data[4, :] = torch.tensor([feature_map_bin[str(int(val.item()))] for val in Train_Dataset.data[4, :]])
Train_Dataset.data[5, :] = torch.tensor([feature_map_bout[str(int(val.item()))] for val in Train_Dataset.data[5, :]])
Train_Dataset.data[8, :] = torch.tensor([feature_map_proto[str(int(val.item()))] for val in Train_Dataset.data[8, :]])

print(len(Train_Dataset.data))
print(len(Test_Dataset.data))

print(f'1: {Train_Dataset.data[4, torch.argmax(Train_Dataset.data[4])]}')
print(f'2: {original_bin[torch.argmax(original_bin)]}')

torch.save(Train_Dataset, U.training_dataset)

### Map Test Values ###
for i, item in enumerate(Test_Dataset.data[4]): #For bin
    Test_Dataset.data[4, i] = preprocessing_functions.approximate_value(item, original_bin)

### Map Test Values ###
for i, item in enumerate(Test_Dataset.data[5]): #For bout
    Test_Dataset.data[5, i] = preprocessing_functions.approximate_value(item, original_bout)

# Explicitly assign back to ensure updates stick
Test_Dataset.data[4, :] = torch.tensor([feature_map_bin[str(int(val.item()))] for val in Test_Dataset.data[4, :]])
Test_Dataset.data[5, :] = torch.tensor([feature_map_bout[str(int(val.item()))] for val in Test_Dataset.data[5, :]])
try:
    Test_Dataset.data[8, :] = torch.tensor([
        feature_map_proto[str(int(val.item()))] 
        for val in Test_Dataset.data[8, :]
    ])
except KeyError as e:
    # Create a new list to store the transformed values
    transformed_values = []
    for val in Test_Dataset.data[8, :]:
        try:
            transformed_values.append(feature_map_proto[str(int(val.item()))])
        except KeyError:
            transformed_values.append(0) #Padding 0 value
    
    # Assign the transformed values back to the dataset
    Test_Dataset.data[8, :] = torch.tensor(transformed_values)

### Initialize Dataloaders ###
Train_Loader = DataLoader(Train_Dataset, batch_size=H.BATCH_SIZE, shuffle=True)
Test_Loader = DataLoader(Test_Dataset, batch_size=H.BATCH_SIZE, shuffle=False)

### Model Initialization ###
############################
### Calculate Embedding Layer Sizes ###
embed_dims = []
num_embeds = []

print("Calculating embedding dims")
for i in range(4, 9):
    print(i)
    feature = Train_Dataset.__getfeature__(i)
    #print(feature[torch.argmin(feature)], feature[torch.argmax(feature)])
    num_features = feature[torch.argmax(feature)].item()
    print(num_features)

    embed_dims.append(int(math.floor(math.sqrt(num_features+1))))
    num_embeds.append(int(num_features)+1)

print("Embedding parameters")
print(f'embed_dims: {embed_dims}')
print(f'num_embeds: {num_embeds}')

model = Model.Network_Dection_Model(H.OUTPUT_DIM, #'avg_ipt', 'entropy', 'total_entropy', 'duration', 1 output neuron
                                    embed_dims[0], num_embeds[0], #Bin
                                    embed_dims[1], num_embeds[1], #Bout
                                    embed_dims[2], num_embeds[2], #Pin
                                    embed_dims[3], num_embeds[3], #Pout
                                    embed_dims[4], num_embeds[4]) #Proto
model.to(device)
#Save model parameters for loading before inference
model_params = {'output': H.OUTPUT_DIM, 
                'embed_dims': embed_dims,
                'num_embeds': num_embeds,
                'original_bin': original_bin.tolist(),
                'original_bout': original_bout.tolist(),
                'feature_map_bin': feature_map_bin,
                'feature_map_bout': feature_map_bout,
                'feature_map_proto': feature_map_proto}

with open(U.model_params, 'w') as j: #Save model parameters
    json.dump(model_params, j, indent=4)

loss_fn = nn.CrossEntropyLoss() #Binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=H.LEARNING_RATE)

#Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

#Training Loop
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss as infinity
epochs_no_improve = 0

percentage_update = .25
update = int(len(Train_Loader) * percentage_update)
amt = 0



for epoch in range(H.EPOCHS):
    print(f"=== EPOCH {epoch + 1} ===")
    model.train()
    training_loss = 0 #Track training loss across that batches
    print("Running Training Loop")
    for i, batch in enumerate(Train_Loader):
        #Forward Pass
        x = batch[:, :9]
        y = batch[:, 9].to(torch.long)
        #Send to device
        x = x.to(device)
        y = y.to(device)

        y_preds = model(x)
        y_preds = y_preds.squeeze(1)

        #Loss
        loss = loss_fn(y_preds, y)
        training_loss += loss.item()

        #Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%update == 0:
            amt += 1
            #print(f'{percentage_update*amt}%')

    training_loss /= len(Train_Loader)
    train_losses.append(training_loss)

    testing_loss, test_acc = 0, 0
    print("Testing the model...")
    model.eval()
    for batch in Test_Loader:
        #Forward Pass
        x = batch[:, :9]
        y = batch[:, 9].to(torch.long)

        #Send to device
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_preds = model(x)

        y_preds = y_preds.squeeze(1)

        #Loss
        loss = loss_fn(y_preds, y)
        testing_loss += loss.item()

        test_acc += accuracy_fn(y_true=y, y_pred=y_preds.argmax(dim=1))
    
    testing_loss /= len(Test_Loader)
    test_acc /= len(Test_Loader)
    val_losses.append(testing_loss)
    print(f"Train loss: {training_loss:.6f} | Test loss: {testing_loss:.6f} | Test acc: {test_acc:.6f}%")

    #Evaluate model
    if testing_loss < best_val_loss:
        best_val_loss = testing_loss
        # Save the model's parameters (state_dict) to a file
        torch.save(model.state_dict(), (U.MODEL_FOLDER / (H.MODEL_NAME + '.pth')).resolve())
        with open((U.MODEL_FOLDER / (H.MODEL_NAME + '_loss.txt')).resolve(), 'w') as f:
            f.write(str(testing_loss))
        print(f'Saved best model with validation loss: {best_val_loss:.6f}')
        epochs_no_improve = 0  # Reset counter if improvement
    else:
        epochs_no_improve += 1
        print(f'Num epochs since improvement: {epochs_no_improve}')

        #stop training if overfitting starts to happen
        if epochs_no_improve >= H.PATIENCE:
            print("Early stopping")
            break

for x in range(H.PATIENCE):
    train_losses.pop(-1)
    val_losses.pop(-1)

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()