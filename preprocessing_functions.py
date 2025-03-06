import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import HyperParameters as H
import Utils as U
import Data_cleanup

def map_values(training_dataset):
    feature_map_bin, feature_map_bout, feature_map_proto = {}, {}, {}

    original_bin = training_dataset.data[4].clone()
    original_bout = training_dataset.data[5].clone()

    bin_set = sorted(list(set(training_dataset.data[4].tolist())))
    for i, item in enumerate(bin_set):
        feature_map_bin[f'{int(item)}'] = i

    bout_set = sorted(list(set(training_dataset.data[5].tolist())))
    for i, item in enumerate(bout_set):
        feature_map_bout[f'{int(item)}'] = i

    proto_set = sorted(list(set(training_dataset.data[8].tolist())))
    feature_map_proto['0'] = 0
    for i, item in enumerate(proto_set):
        feature_map_proto[f'{int(item)}'] = i + 1
    
    return feature_map_bin, feature_map_bout, feature_map_proto, original_bin, original_bout

def approximate_value(item, original):
    # Calculate absolute differences
    differences = torch.abs(original - item)

    # Get the index of the minimum difference
    closest_index = torch.argmin(differences).item()

    return original[closest_index]
    