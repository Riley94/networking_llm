import os
import pandas as pd
import kagglehub
from tqdm import tqdm

from data_loading.tools import reduce_mem_usage

#import gdown
from tqdm import tqdm
#from googleapiclient.discovery import build
#from googleapiclient.http import MediaFileUpload
#from google.oauth2 import service_account
import tempfile
import shutil

# Set your Google Drive Folder ID where the combined file will be stored
GDRIVE_FOLDER_ID = "luflow_full"

def authenticate_drive():
    """Authenticate and return the Google Drive service."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    creds_path = os.path.join(script_dir, "credentials.json")  # Full path to credentials.json
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def check_drive_file_exists(service, filename):
    """Check if the file already exists in Google Drive."""
    try:
        query = f"name='{filename}' and '{GDRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        return files[0]["id"]

    except Exception as e:
        print(f"Error checking if file exists: {e}")
        return None
    

def download_from_drive(service, file_id, save_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    with open(save_path, "wb") as f:
        request.execute().write_to(f)
    print(f"Downloaded {save_path} from Google Drive.")


def combine_luflow(data_path, save_path, chunk_size=100_000):
    """Combine the LUFlow dataset into a single CSV file."""
    # Use a temporary file to avoid loading everything into memory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "temp_combined.csv")
    
    first_write = True
    file_count = 0
    row_count = 0

    try:
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
                                        file_count += 1
                                        full_path = os.path.join(day_path, file)

                                        # Read in chunks (adjust chunksize as needed)
                                        for chunk in pd.read_csv(full_path, chunksize=chunk_size):
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

                                # Write to temp file
                                chunk.to_csv(temp_file, mode='a', header=first_write, index=False)
                                first_write = False
                                row_count += len(chunk)
                                
                                # Periodically report progress
                                if row_count % (chunk_size * 10) == 0:
                                    print(f"Processed {row_count} rows so far...")
            
            # Copy the temp file to the final destination
            shutil.copy2(temp_file, save_path)
            print(f"Finished merging {file_count} CSV files ({row_count} total rows) into {save_path}")
            
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

def get_luflow(num_rows=750_000, process_local=True, overwrite=False):
    # Path to the cached dataset
    cache_path = os.path.expanduser("~/.cache/kagglehub/datasets")
    data_path = os.path.join(cache_path, "mryanm/luflow-network-intrusion-detection-data-set/versions/240")
    dir = os.path.dirname(os.path.realpath(__file__))
    
    # get parent directory of dir
    src_dir = os.path.join(dir, os.pardir)
    combined_data_dir = os.path.join(src_dir, os.pardir, "data")
    filename = "luflow_combined.csv"
    combined_data_path = os.path.join(combined_data_dir, filename)
    os.makedirs(combined_data_dir, exist_ok=True)

    if not process_local:
        # Authenticate with Google Drive
        service = authenticate_drive()

        # Check if the file exists on Google Drive
        file_id = check_drive_file_exists(service, filename)

        if file_id:
            # Download the file from Google Drive
            download_from_drive(service, file_id, combined_data_path)
    else:
        if not (os.path.exists(data_path) or os.path.exists(combined_data_path)) or overwrite: # if either of the paths exist, don't download
            print("Downloading LUFlow dataset...")
            # Download latest version
            data_path = kagglehub.dataset_download("mryanm/luflow-network-intrusion-detection-data-set")

        if not os.path.exists(combined_data_path) or overwrite:
            print("Combining LUFlow dataset...")
            combine_luflow(data_path, combined_data_path)
            # remove cache data_path
            os.system(f"rm -rf {data_path}")

    print("Loading combined LUFlow dataset...")
    data = pd.read_csv(combined_data_path, nrows=num_rows)
    data.drop(['src_ip', 'dest_ip', 'time_start', 'time_end', 'label'], axis=1, inplace=True)
    data.dest_port = data.dest_port.fillna(-1).astype('int64')
    data.src_port = data.src_port.fillna(-1).astype('int64')
    data = reduce_mem_usage(data)
    print(f"Columns: {data.columns}")
    return data.to_numpy()