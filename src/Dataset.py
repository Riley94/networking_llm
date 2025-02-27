import torch
from torch.utils.data import Dataset

class LUFlow_ND_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, index):
        return self.data[:, index]
    
    def __getfeature__(self, index):
        return self.data[index]