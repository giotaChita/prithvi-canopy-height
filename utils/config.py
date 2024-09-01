import torch
import datetime
import os

checkpoint_path = os.getcwd() + '/model/save_load_model/checkpoint.pth'

# to save gedi shots with good quality
out_path_quality_shots_rh98 = os.getcwd() + '/data/gedi_swiss_qualityshots_aoi_rh98.json'
out_path_quality_shots_rh99 = os.getcwd() + '/data/gedi_swiss_qualityshots_aoi_rh99.json'
out_path_quality_shots2 = os.getcwd() + '/data/gedi_swiss_qualityshots_aoi2.json'
out_path_quality_shots1 = os.getcwd() + '/data/gedi_swiss_qualityshots_aoi1.json'

# path to gedi shots
cache_path = os.getcwd()+"/data/canopy_height_labels.npy"
cache_path_rh98 = os.getcwd() + "/data/canopy_height_labels_rh98.npy"
cache_path_rh99 = os.getcwd() + "/data/canopy_height_labels_rh99.npy"


# Model pretrained w
pretrained_model_path = os.getcwd() + '/prithvi/Prithvi_100M.pt'

# Best model path
# Get the current date and time
now = datetime.datetime.now()

# Format the date and time as YYYY_MM_DD_HHMMSS
timestamp = now.strftime('%Y_%m_%d_%H%M%S')

# Define the base path and filename
base_path = os.getcwd() + '/model/save_load_model/'
filename = f'best_model_state_{timestamp}.pth'

# Construct the full path
best_model_path_new = f'{base_path}{filename}'

# Best Model Path
best_model_path = os.getcwd() + '/model/save_load_model/best_model_state_New.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to gedi shots for aoi1
gedi_path1 = os.getcwd() + "/data/gedi_data_aoi1"

# Path to gedi shots for aoi2 (bigger)
gedi_path2 = os.getcwd() + "/data/gedi_data_aoi2"

