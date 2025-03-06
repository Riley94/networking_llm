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
    print("Loading data...")
    try:
        df = pd.read_parquet(U.raw_data)
        # Detailed min and max with column names
        for column in df.columns:
            print(f"{column}:")
            print(f"  Min: {df[column].min()}")
            print(f"  Max: {df[column].max()}")
        df_2 = pd.read_parquet(U.raw_data_2)
        for column in df.columns:
            print(f"{column}:")
            print(f"  Min: {df[column].min()}")
            print(f"  Max: {df[column].max()}")
        df_3 = pd.read_parquet(U.raw_data_3)
        for column in df.columns:
            print(f"{column}:")
            print(f"  Min: {df[column].min()}")
            print(f"  Max: {df[column].max()}")
    except: 
        import requests

        url = "https://your-file-hosting.com/dataset.zip"  # Replace with actual URL
        output_path = "dataset.zip"

        response = requests.get(url, stream=True)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Dataset downloaded successfully!")
    print("Concatenating data...")
    df = pd.concat([df, df_2, df_3], ignore_index=True)

    # Drop rows with any null values
    df = df.dropna()

    # Print unique values in the 'label' column
    print("Unique labels:", df['label'].unique())

    print(df.head())
    print(set(df['label'].values))
    return df

def balance_classes(df):
    # Get the minimum count of any label
    min_count = df['label'].value_counts().min()

    # Sample 'min_count' rows from each class
    df_balanced = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

    return df_balanced

def preprocess_data(df):
    '''columns ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out',
       'num_pkts_in', 'proto', 'total_entropy', 'duration', 'label',
       'creation_date']'''
    df = df.drop(['creation_date'], axis=1)

    '''cols = df.columns
    original_rows = len(df)
    df = df[~df[cols].eq(-1).any(axis=1)]
    print(f"Rows removed: {original_rows - len(df)}")'''

    label_map = {'benign': 0, 'outlier': 1, 'malicious': 2}
    df['label'] = df['label'].map(label_map) #map labels to numbers

    #Balance classes
    print("Balancing class labels...")
    df = balance_classes(df)

    #Rearrange dataframe
    #Continuous data = indexes [0:4] (first 4 indexes)
    df = df[['avg_ipt', 'entropy', 'total_entropy', 'duration', 'bytes_in', 'bytes_out', 'num_pkts_in', 'num_pkts_out', 'proto', 'label']]
    cols = df.columns
    print(df.columns)
    print(df.head())
    

    data = np.zeros((df.shape[1], df.shape[0])) #Initialize np array of the df size

    print(f"Unique labels: {df['label'].unique()}")

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

    for column in df.columns:
        print(f"{column}:")
        print(f"  Min: {df[column].min()}")
        print(f"  Max: {df[column].max()}")

    data = torch.zeros(df.shape[1], df.shape[0])
    for i, col in enumerate(df.columns):
        data[i] = torch.tensor(df[col].values, dtype=torch.float32)  # Convert to tensor first

    print(data[0])

    #data = torch.from_numpy(data)
    print(torch.unique(data[:, -1]))
    print(data[:5, :])
    print(data[-5:, :])

    for x in range(len(data)):
        check_feature = x
        print(f'min col {cols[x]}: {data[check_feature, torch.argmin(data[check_feature])]}, max: {data[check_feature, torch.argmax(data[check_feature])]}')

    return data

def clean_data():
    df = load_data()
    data = preprocess_data(df)
    torch.save(data, U.clean_data)
    print("Data saved, cleanup finished.")

if __name__ == "__main__":
    clean_data()
