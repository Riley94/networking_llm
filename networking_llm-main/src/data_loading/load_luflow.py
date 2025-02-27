import os
import pandas as pd
import kagglehub
from tqdm import tqdm

from data_loading.tools import reduce_mem_usage

def combine_luflow(data_path, save_path):
    first_write = True  # Track if it's the first write to include headers

    # Load and combine each CSV
    for year in tqdm(sorted(os.listdir(data_path))):  # Sorting for consistency
        year_path = os.path.join(data_path, year)
        if os.path.isdir(year_path):
            for month in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    for day in sorted(os.listdir(month_path)):
                        day_path = os.path.join(month_path, day)
                        if os.path.isdir(day_path):
                            for file in os.listdir(day_path):
                                if file.endswith(".csv"):
                                    full_path = os.path.join(day_path, file)

                                    # Read in chunks (adjust chunksize as needed)
                                    for chunk in pd.read_csv(full_path, chunksize=10_000):
                                        # Extract date info
                                        try:
                                            y, m, d = map(int, file.split(".")[:3])
                                            chunk["Year"] = y
                                            chunk["Month"] = m
                                            chunk["Day"] = d
                                            chunk['label'] = pd.Categorical(chunk['label']).codes
                                        except ValueError:
                                            print(f"Skipping malformed filename: {file}")
                                            continue

                                        # Append to file (write header only once)
                                        chunk.to_csv(save_path, mode='a', header=first_write, index=False)
                                        first_write = False  # Only write header in the first batch

    print(f"Finished merging CSVs into {save_path}")

def get_luflow():
    OVERWRITE = False

    # Path to the cached dataset
    cache_path = os.path.expanduser("~/.cache/kagglehub/datasets")
    data_path = os.path.join(cache_path, "mryanm/luflow-network-intrusion-detection-data-set/versions/240")
    dir = os.path.dirname(os.path.realpath(__file__))
    # get parent directory of dir
    src_dir = os.path.join(dir, os.pardir)
    combined_data_path = os.path.join(src_dir, os.pardir, "data", "luflow_combined.csv")

    if not (os.path.exists(data_path) or os.path.exists(combined_data_path)) or OVERWRITE: # if either of the paths exist, don't download
        # Download latest version
        data_path = kagglehub.dataset_download("mryanm/luflow-network-intrusion-detection-data-set")

    if not os.path.exists(combined_data_path) or OVERWRITE:
        combine_luflow(data_path, combined_data_path)
        # remove cache data_path
        os.system(f"rm -rf {data_path}")

    data = pd.read_csv(combined_data_path, nrows=750_000)
    data.drop(['src_ip', 'dest_ip', 'time_start', 'time_end', 'label'], axis=1, inplace=True)
    data.dest_port = data.dest_port.fillna(-1).astype('int64')
    data.src_port = data.src_port.fillna(-1).astype('int64')
    data = reduce_mem_usage(data)
    return data.to_numpy()