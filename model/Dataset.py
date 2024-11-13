import os
import torch.nn.functional as F
import numpy as np
from osgeo import gdal
from utils.utils import preprocess_image
from utils.gedi_data import load_gedi_shots
from torch.utils.data import Dataset
from utils.utils import latlon_to_pixel
from utils.config import (cache_path_rh98, gedi_path2, gedi_path1, cache_path_rh99, cache_path_aoi3_gedi_shots_rh98, cache_path_aoi3_rh95_filter3, cache_path_aoi4_rh95_second_filter_waterzero,
                          cache_path_aoi4_rh95_new_filter, cache_path_aoi4_rh95_filter3, gedi_path5, cache_path_aoi4_rh95_old_filter, cache_path_aoi4_rh95_second_filter,
                          gedi_path3, cache_path_aoi3_gedi_shots_rh98_size1024, gedi_path4, cache_path_aoi4_gedi_shots_rh98,cache_path,cache_path_aoi4_gedi_shots_rh98_28_9, cache_path_aoi5_rh95)
import torch
from matplotlib import pyplot as plt
import pandas as pd
from pyproj import Proj, transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.stats import multivariate_normal
from skimage.draw import disk
import random


class TiledDataset(Dataset):
    def __init__(self, image_array_list, label_array, tile_size, overlap, padding=0):
        self.image_array_list = image_array_list
        self.label_array = label_array
        self.tile_size = tile_size
        self.overlap = overlap
        self.padding = padding
        self.tiles = self.generate_tiles()
        # print("ok")

    def generate_tiles(self):
        step_size = self.tile_size - self.overlap
        image_height, image_width = self.image_array_list[0].shape[1], self.image_array_list[0].shape[2]

        # start points of the tiles
        row_starts = np.arange(0, image_height + 1, step_size)
        col_starts = np.arange(0, image_width + 1, step_size)
        # meshgrid of the start points
        grid_r, grid_c = np.meshgrid(row_starts, col_starts, indexing='ij')
        tile_start_points = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1)

        return tile_start_points

    def pad_tile(self, tile, tile_height, tile_width, labels=False):
        """
        Pads the input tile to the required tile_size (self.tile_size) with zeros or NaNs (for labels).
        """
        pad_h = self.tile_size - tile_height
        pad_w = self.tile_size - tile_width

        pad_value = float('nan') if labels else self.padding

        if pad_h > 0 or pad_w > 0:
            padding = (0, pad_w, 0, pad_h)

            tile = F.pad(tile, padding, value=pad_value)

        return tile

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        start_row, start_col = self.tiles[idx]
        start_row, start_col = int(start_row), int(start_col)
        end_row, end_col = int(start_row + self.tile_size), int(start_col + self.tile_size)

        img_list = []
        for image_array in self.image_array_list:
            image_tile = image_array[:, start_row:end_row, start_col:end_col]
            image_tile_height, image_tile_width = image_tile.shape[-2], image_tile.shape[-1]
            if image_tile_height != self.tile_size or image_tile_width != self.tile_size:
                image_tile = self.pad_tile(image_tile, image_tile_height, image_tile_width)
            img_list.append(image_tile)

        img = torch.stack(img_list, dim=1)

        label_tile = self.label_array[start_row:end_row, start_col:end_col]
        label_tile_height, label_tile_width = label_tile.shape[-2], label_tile.shape[-1]

        if label_tile_height != self.tile_size or label_tile_width != self.tile_size:
            label_tile = self.pad_tile(label_tile, label_tile_height, label_tile_width, labels=True)

        lbl = torch.stack([label_tile, label_tile], dim=0)

        return img, lbl

    def count_non_nan_pixels(self):
        valid_pixels = torch.sum(~torch.isnan(self.label_array))
        total_pixels = self.label_array.numel()
        return valid_pixels.item(), total_pixels


class TiledDatasetNoPadding(Dataset):
    def __init__(self, image_array_list, label_array, tile_size, overlap):
        self.image_array_list = image_array_list
        self.label_array = label_array
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles = self.generate_tiles()
        self.time_dimension = len(image_array_list)
    def generate_tiles(self):
        step_size = self.tile_size - self.overlap
        image_height, image_width = self.image_array_list[0].shape[1], self.image_array_list[0].shape[2]

        row_starts = np.arange(0, image_height, step_size)
        col_starts = np.arange(0, image_width, step_size)
        tile_start_points = []

        for r in row_starts:
            for c in col_starts:
                # Check if the tile is within bounds
                if r + self.tile_size <= image_height and c + self.tile_size <= image_width:
                    tile_start_points.append((r, c))

        return tile_start_points

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        start_row, start_col = self.tiles[idx]
        end_row, end_col = start_row + self.tile_size, start_col + self.tile_size

        img_list = []
        for image_array in self.image_array_list:
            image_tile = image_array[:, start_row:end_row, start_col:end_col]
            img_list.append(image_tile)

        img = torch.stack(img_list, dim=1)

        label_tile = self.label_array[start_row:end_row, start_col:end_col]
        lbl = torch.stack([label_tile] * self.time_dimension, dim=0)

        return img, lbl


class AugmentedTiledDataset(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

        # Select half of the indices for augmentation
        self.augmentation_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 2)

        # Define augmentation transforms
        self.augmentation_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        # Return the number of augmented samples
        return len(self.augmentation_indices)

    def __getitem__(self, idx):
        # Get the original index for the selected augmented dataset
        original_index = self.augmentation_indices[idx]

        # Get original image and label
        image, label = self.train_dataset[original_index]

        # Convert to numpy for augmentation
        image = image.numpy()
        label = label.numpy()
        augmented_images = []
        augmented_labels = []

        # Loop over time
        for t in range(image.shape[1]):
            img_t = image[:,t,:,:].transpose(1, 2, 0)
            lbl_t = label[t,: , :]

            # Apply augmentation
            augmented = self.augmentation_transforms(image=img_t, mask=lbl_t)
            augmented_images.append(augmented['image']) #.permute(2, 0, 1))  # (H, W, C) -> (C, H, W)
            augmented_labels.append(augmented['mask'])

        # Stack along the T dimension
        images = torch.stack(augmented_images, dim=1)  # (C, T, H, W)
        # labels = torch.stack([torch.tensor(l) for l in augmented_labels], dim=0)  # (T, H, W)
        labels = torch.stack([l.clone().detach() for l in augmented_labels], dim=0)  # (T, H, W)

        return images, labels


def canopy_height_GEDI(hls_path, response):
    """
    Load Gedi Shots form 2019 and return canopy height return
    :param hls_path: path from hls image
    :param response: array with hls data
    :return: canopy height labels numpy array (NaN values where there are no gedi shots)
    """

    # load Gedi Shots from year 2019 and aoi the given one

    #Bigger AOI -> Polygon
    gedi_shots = load_gedi_shots(gedi_path4)  #
    # path_dataframe_gedi_aoi5_rh95 = os.getcwd() + '/data/gedi_data_aoi5/gedi_shots_aoi5_rh95.csv'  # aoi5
    # gedi_shots.to_csv(path_dataframe_gedi_aoi5_rh95, index=False)


    # path_dataframe_gedi_aoi4_rh95 = os.getcwd() + '/data/gedi_data_aoi4/gedi_shots_aoi4_rh95_comb3_filter.csv'  # aoi4
    # gedi_shots.to_csv(path_dataframe_gedi_aoi4_rh95, index=False)
    # gedi_shots = pd.read_csv(path_dataframe_gedi_aoi4_rh95)  # aoi4

    # gedi_shots = load_gedi_shots(gedi_path3)  # aoi3
    # path_dataframe_gedi_aoi3 = os.getcwd() + '/data/gedi_data_aoi3/gedi_shots_aoi3_rh98.csv'  # aoi3
    # gedi_shots.to_csv(path_dataframe_gedi_aoi3, index=False)
    # gedi_shots = pd.read_csv(path_dataframe_gedi_aoi3)  # aoi3

    # gedi_shots = load_gedi_shots(gedi_path4)  # aoi4
    # path_dataframe_gedi = os.getcwd() + '/data/gedi_data_aoi4/gedi_shots_aoi4.csv'  # aoi4
    # gedi_shots = pd.read_csv(path_dataframe_gedi)  # aoi4

    ds = gdal.Open(hls_path)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {hls_path}")

    geotransform = ds.GetGeoTransform()  # Get geotransform info for geospatial transformation

    for band_index in range(response.shape[2]):
        band_data = response[:, :, band_index]
        canopy_height_labels = np.full_like(band_data, np.nan, dtype=np.float32)
        # canopy_height_labels = np.zeros_like(band_data, dtype=np.float32)

        # Iterate over GEDI shots and update corresponding region
        for index, row in gedi_shots.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            canopy_height = row['Canopy Height (rh95)']

            pixel_x, pixel_y = latlon_to_pixel(lat, lon, geotransform)

            # canopy_height_labels[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = canopy_height
            if pixel_x >= response.shape[1] or pixel_y >= response.shape[0]:
                print(pixel_x, pixel_y)
                print("x and y is out of boundaries")
            else:
                canopy_height_labels[pixel_y, pixel_x] = canopy_height

    # canopy_height_labels = np.where(canopy_height_labels==0, np.nan, canopy_height_labels)

    np.save(cache_path_aoi4_rh95_second_filter_waterzero, canopy_height_labels)

    return canopy_height_labels, gedi_shots

def canopy_height_GEDI_3(hls_path):
    ds = gdal.Open(hls_path)
    path_dataframe_gedi = os.getcwd() + '/data/gedi_data_aoi4/gedi_shots_aoi4.csv'
    gedi_shots = pd.read_csv(path_dataframe_gedi)

    ds = gdal.Open(hls_path)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {hls_path}")


    hls_width = ds.RasterXSize
    hls_height = ds.RasterYSize
    geotransform = ds.GetGeoTransform()

    # Initialize canopy height labels with NaN values
    canopy_height_labels = np.full((hls_height, hls_width), np.nan, dtype=np.float32)
    geolocation_error = 10.2
    num_samples = 100
    count = 0
    gedi_radius = 12.5
    resolution = 30
    pixel_radius = int(np.ceil(gedi_radius / resolution))
    canopy_height_labels_2 = np.full((hls_height, hls_width), np.nan, dtype=np.float32)

    # Iterate over GEDI shots
    for index, row in gedi_shots.iterrows():
        count +=1
        lat = row['Latitude']
        lon = row['Longitude']
        canopy_height = row['Canopy Height (rh98)']

        pixel_x, pixel_y = latlon_to_pixel(lat, lon, geotransform)
        # Generate random samples based on geolocation error (Gaussian distribution)
        cov_matrix = np.array([[geolocation_error**2, 0], [0, geolocation_error**2]])  # Covariance matrix for sampling
        mean = [pixel_x, pixel_y]

        # Randomly sample possible GEDI shot locations
        samples = multivariate_normal(mean, cov_matrix).rvs(num_samples)

        pixel_freq = {}

        # Round to pixel coordinates and apply to the labels
        for px, py in samples:
            px = int(px)
            py = int(py)
            if 0 <= px < hls_width and 0 <= py < hls_height:
                if (px, py) not in pixel_freq:
                    pixel_freq[(px, py)] = 1
                else:
                    pixel_freq[(px, py)] += 1

        if pixel_freq:
            central_pixel = max(pixel_freq, key=pixel_freq.get)
            rr, cc = disk((central_pixel[1], central_pixel[0]), pixel_radius, shape=canopy_height_labels.shape)
            canopy_height_labels_2[rr, cc] = canopy_height
            canopy_height_labels[central_pixel[1], central_pixel[0]] = canopy_height

    return canopy_height_labels, canopy_height_labels_2


def preprocess_data(response):
    hls_data = torch.tensor(response, dtype=torch.float32)
    hls_data = hls_data.permute(2, 0, 1)

    response = np.transpose(response, (2, 0, 1))

    #  mean and std for each band
    means = torch.mean(hls_data, dim=(1, 2))
    stds = torch.std(hls_data, dim=(1, 2))

    means = means.numpy().reshape(-1, 1, 1)
    stds = stds.numpy().reshape(-1, 1, 1)

    hls_img = preprocess_image(response, means, stds)

    hls_data = hls_img.clone().detach()  # shape (1,6,1,H,W)
    hls_data = hls_data.squeeze(0)  # shape (6,1,H,W)
    hls_data = hls_data.squeeze(1)  # shape (6,H,W)

    return hls_data, response
