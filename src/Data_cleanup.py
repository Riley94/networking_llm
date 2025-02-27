import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import HyperParameters as H
import Utils as U
import pickle

def load_data():
    #Download dataset if not already had
    try:
        df = pd.read_parquet(U.raw_data)
    except: 
        import requests

        url = "https://your-file-hosting.com/dataset.zip"  # Replace with actual URL
        output_path = "dataset.zip"

        response = requests.get(url, stream=True)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Dataset downloaded successfully!")

    print(df.head())
    print(set(df['label'].values))
    return df

def preprocess_data(df):
    '''columns ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out',
       'num_pkts_in', 'proto', 'total_entropy', 'duration', 'label',
       'creation_date']'''
    df = df.drop(['creation_date'], axis=1)

    label_map = {'benign': 0, 'outlier': 1, 'malicious': 2}
    df['label'] = df['label'].map(label_map) #map labels to numbers

    # Calculate counts and percentages
    label_counts = df['label'].value_counts()
    label_percentages = label_counts / len(df) * 100

    # Print counts and percentages
    print("Label Distribution:")
    for label, count in label_counts.items():
        percentage = label_percentages[label]
        print(f"{label}: {count} instances ({percentage:.2f}%)")

    
    # Create figure
    plt.figure(figsize=(12, 6))
    ax = label_counts.plot(kind='bar', color='steelblue')

    # Primary y-axis: counts
    plt.ylabel('Count')
    plt.title('Distribution of Labels: Count and Percentage')

    # Add count labels
    for i, count in enumerate(label_counts.values):
        plt.text(i, count + (max(label_counts.values) * 0.02), 
                f"{count}\n({label_percentages[label_counts.index[i]]:.1f}%)", 
                ha='center')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #Rearrange dataframe
    #Continuous data = indexes [0:4] (first 4 indexes)
    df = df[['avg_ipt', 'entropy', 'total_entropy', 'duration', 'bytes_in', 'bytes_out', 'num_pkts_in', 'num_pkts_out', 'proto', 'label']]

    print(df.columns)
    print(df.head())
    

    data = np.zeros((df.shape[1], df.shape[0])) #Initialize np array of the df size

    #Extract all values from the dataframe into the data array
    for index, col in enumerate(df.columns):
        data[index] = df[col].values

    data = torch.from_numpy(data)

    return data

def clean_data():
    df = load_data()
    data = preprocess_data(df)
    torch.save(data, U.clean_data)

if __name__ == "__main__":
    clean_data()
