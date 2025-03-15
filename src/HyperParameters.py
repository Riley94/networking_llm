import torch

### Other Hyperparameters ###
train_test_split = .8
continuous_variables = ['avg_ipt', 'entropy', 'total_entropy', 'duration']
CLASSES = ['Benign', 'Outlier', 'Malicious']

### MODEL HYPER PARAMETERS ###
#Model name
MODEL_NAME = 'Model_0'
BATCH_SIZE = 16
LEARNING_RATE = .0001
EPOCHS = 50
PATIENCE = 3
OUTPUT_DIM = 3

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"