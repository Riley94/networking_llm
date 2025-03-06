from pathlib import Path
import HyperParameters
import Utils as U

#Dataset location information 
PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DATA_FOLDER = (PROJECT_DIR / 'data_clean').resolve()
RAW_DATA_FOLDER = (PROJECT_DIR / 'data_raw').resolve()
TEST_DATA_FOLDER = (PROJECT_DIR / 'data_test').resolve()
MODEL_FOLDER = (PROJECT_DIR / 'Model').resolve()
SCALERS_FOLDER = (PROJECT_DIR / 'Scalers').resolve()

raw_data = (U.RAW_DATA_FOLDER / 'LUFlow-2022.parquet').resolve()
raw_data_2 = (U.RAW_DATA_FOLDER / 'LUFlow-2021.parquet').resolve()
raw_data_3 = (U.RAW_DATA_FOLDER / 'LUFlow-2020.parquet').resolve()
clean_data = (U.CLEAN_DATA_FOLDER / 'cleaned_data.pt').resolve()
training_dataset = (U.CLEAN_DATA_FOLDER / 'training_dataset.pt').resolve()
feature_map_bin = (U.CLEAN_DATA_FOLDER / 'bin_map.json').resolve()
feature_map_bout = (U.CLEAN_DATA_FOLDER / 'bout_map.json').resolve()
feature_map_proto = (U.CLEAN_DATA_FOLDER / 'proto_map.json').resolve()
model_params = (U.CLEAN_DATA_FOLDER / 'model_params.json').resolve()
test_data = (U.CLEAN_DATA_FOLDER / 'test_data.pt').resolve()