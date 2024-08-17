import numpy as np
from osgeo import gdal
from utils.utils import preprocess_image
from utils.gedi_data import load_gedi_shots
from torch.utils.data import Dataset
from utils.utils import latlon_to_pixel
from utils.config import cache_path, gedi_path2, gedi_path1
import torch


class TiledDataset(Dataset):
    def __init__(self, image_array, label_array, tile_size, overlap):
        self.image_array = image_array
        self.label_array = label_array
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles = self.generate_tiles()
        # print("ok")

    def generate_tiles(self):
        step_size = self.tile_size - self.overlap
        image_height, image_width = self.image_array.shape[1], self.image_array.shape[2]

        # Calculate the start points of the tiles
        row_starts = np.arange(0, image_height - self.tile_size + 1, step_size)
        col_starts = np.arange(0, image_width - self.tile_size + 1, step_size)

        # Ensure the last tile reaches the edge of the image
        if (image_height - self.tile_size) % step_size != 0:
            row_starts = np.append(row_starts, image_height - self.tile_size)
        if (image_width - self.tile_size) % step_size != 0:
            col_starts = np.append(col_starts, image_width - self.tile_size)

        # Create a meshgrid of the start points
        grid_r, grid_c = np.meshgrid(row_starts, col_starts, indexing='ij')
        tile_start_points = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1)

        return tile_start_points

    def __len__(self):
        # print(len(self.tiles))
        return len(self.tiles)

    def __getitem__(self, idx):
        start_row, start_col = self.tiles[idx]
        start_row = int(start_row)
        start_col = int(start_col)
        end_row = int(start_row + self.tile_size)
        end_col = int(start_col + self.tile_size)

        image_tile = self.image_array[:, start_row:end_row, start_col:end_col]
        label_tile = self.label_array[start_row:end_row, start_col:end_col]

        img = image_tile.clone()
        lbl =label_tile.clone()

        img = img.unsqueeze(1)
        lbl = lbl.unsqueeze(0)

        return img, lbl

    def count_non_nan_pixels(self):
        valid_pixels = torch.sum(~torch.isnan(self.label_array))
        total_pixels = self.label_array.numel()
        return valid_pixels.item(), total_pixels


def canopy_height_GEDI(hls_path, response):
    """
    Load Gedi Shots form 2019 and return canopy height return
    :param hls_path: path from hls image
    :param response: array with hls data
    :return: canopy height labels numpy array (NaN values where there are no gedi shots)
    """

    # load Gedi Shots from year 2019 and aoi the given one

    #Bigger AOI -> Polygon
    gedi_shots = load_gedi_shots(gedi_path2)

    ds = gdal.Open(hls_path)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {hls_path}")

    geotransform = ds.GetGeoTransform()  # Get geotransform info for geospatial transformation

    for band_index in range(response.shape[2]):
        band_data = response[:, :, band_index]
        canopy_height_labels = np.zeros_like(band_data, dtype=np.float32)

        # Parameters for the GEDI footprint
        footprint_radius = 25 / 2  # Radius of 12.5 meters
        resolution = 30
        pixel_radius = int(np.ceil(footprint_radius / resolution))  # Convert radius to pixels

        # Iterate over GEDI shots and update corresponding region
        for index, row in gedi_shots.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            canopy_height = row['Canopy Height (rh100)']

            # Convert latitude and longitude to pixel coordinates in  image
            pixel_x, pixel_y = latlon_to_pixel(lat, lon, geotransform)
            # Define the region around the pixel coordinates
            patch_x_start = max(0, pixel_x)
            patch_x_end = min(band_data.shape[1], pixel_x)
            patch_y_start = max(0, pixel_y )
            patch_y_end = min(band_data.shape[0], pixel_y)

            # canopy_height_labels[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = canopy_height
            canopy_height_labels[pixel_y, pixel_x] = canopy_height

    canopy_height_labels = np.where(canopy_height_labels==0, np.nan, canopy_height_labels)
    np.save(cache_path, canopy_height_labels)
    return canopy_height_labels, gedi_shots


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

    hls_data = hls_img.clone().detach()  # shape (1,6,1,512,512)
    hls_data = hls_data.squeeze(0)  # shape (6,1,512,512)
    hls_data = hls_data.squeeze(1)  # shape (6,512,512)

    return hls_data, response
