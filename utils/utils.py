from sentinelhub import SHConfig, DataCollection, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, MosaickingOrder
from datetime import datetime, timedelta
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import os, io, tarfile, json
import shapely.wkt
from osgeo import gdal
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
import rasterio
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

gdal.UseExceptions()

CLIENT_ID = "744033d4-6720-4a67-89e7-ffda2113f57c"
CLIENT_SECRET = "qoyB4xFJD1xGJIFATEuZNaxHUZRwOSuT"

CLIENT_ID_COP = "sh-4b71279f-c0a6-4692-bc42-59cf99ac8df0"
CLIENT_SECRET_COP = "Pejh6vTsk6y9NwLekYK8ilFLdZUpsnqK"

GDAL_TYPE_LOOKUP = {'float32': gdal.GDT_Float32,
                    'float64': gdal.GDT_Float64,
                    'uint16': gdal.GDT_UInt16,
                    'uint8': gdal.GDT_Byte}

def load_model_with_checkpoint(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Print checkpoint keys for debugging
    print("Checkpoint keys:", checkpoint.keys())

    # Manually load parameters into the model
    model_params = model.state_dict()

    for key in checkpoint.keys():
        if key in model_params:
            # Update model parameters with checkpoint values
            model_params[key].copy_(checkpoint[key])
        else:
            print(f"Skipping key {key} as it is not found in the model parameters.")

    # Update model state dict
    model.load_state_dict(model_params, strict=False)

    return model


def plot_loader_images(loader, loader_name):
    for images, labels in loader:
        print(f"{loader_name} images shape:", images.shape)  # (batch_size, C, T, tile_size, tile_size)
        print(f"{loader_name} labels shape:", labels.shape)  # (batch_size, T, tile_size, tile_size)

        # Convert to numpy for plotting
        img = images[0].squeeze(1).permute(1, 2, 0).numpy()  # Select the first image and change the order to (H, W, C)
        lbl = labels[0].squeeze(0).numpy()  # Select the first label

        # Plot the image and label
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img[:, :, 0], cmap='gray')  # Assuming grayscale images
        axs[0].set_title('Image Tile - Channel 0')
        axs[0].axis('off')

        cax = axs[1].imshow(lbl, cmap='viridis')
        axs[1].set_title('Label Tile')
        axs[1].axis('off')
        fig.colorbar(cax, ax=axs[1], fraction=0.046, pad=0.04)

        plt.show()
        break


def plot_loaders(train_loader, test_loader, val_loader):
    plot_loader_images(train_loader, "Train")
    plot_loader_images(test_loader, "Test")
    plot_loader_images(val_loader, "Validation")


def latlon_to_pixel(lat, lon, geo_transform):
    """
    Convert latitude and longitude to pixel coordinates using the provided geotransform parameters.

    Parameters:
    - lat: Latitude in decimal degrees
    - lon: Longitude in decimal degrees
    - geo_transform: Tuple containing geotransform parameters (geotransform[0] - top left x,
                     geotransform[1] - w-e pixel resolution, geotransform[2] - rotation,
                     geotransform[3] - top left y, geotransform[4] - rotation, geotransform[5] - n-s pixel resolution)

    Returns:
    - pixel_x: Pixel x coordinate
    - pixel_y: Pixel y coordinate
    """
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    pixel_x = int((lon - x_origin) / pixel_width)
    pixel_y = int((lat - y_origin) / pixel_height)

    return pixel_x, pixel_y


def compute_metrics(predictions, targets):
    # Flatten predictions and targets
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Handle NaN values in targets
    nan_mask = ~np.isnan(targets_flat)
    predictions_flat = predictions_flat[nan_mask]
    targets_flat = targets_flat[nan_mask]

    # Compute metrics
    mae = mean_absolute_error(targets_flat, predictions_flat)
    rmse = mean_squared_error(targets_flat, predictions_flat, squared=False)
    r2 = r2_score(targets_flat, predictions_flat)

    return mae, rmse, r2

# def load_tif_as_array(path, set_nodata_to_nan=True, dtype=float):
#     ds = gdal.Open(path)
#     band = ds.GetRasterBand(1)
#
#     array = band.ReadAsArray().astype(dtype)
#     tile_info = get_tile_info(ds)
#     # set the nodata values to nan
#     nodata_value = band.GetNoDataValue()
#     tile_info['nodata_value'] = nodata_value
#     if set_nodata_to_nan:
#         array[array == nodata_value] = np.nan
#     return array, tile_info

def load_tif_as_array(path, set_nodata_to_nan=True, dtype=float):
    ds = gdal.Open(path)
    num_bands = ds.RasterCount
    h = ds.RasterXSize
    w = ds.RasterYSize

    array = np.zeros((h, w, num_bands), dtype=dtype)
    tile_info = get_tile_info(ds)
    no_data = []

    for i in range(1,num_bands+1):
        band= ds.GetRasterBand(i)
        band_arr = band.ReadAsArray().astype(dtype)
        nodata_value = band.GetNoDataValue()
        no_data.append(nodata_value)

        if set_nodata_to_nan and nodata_value is not None:
            band_arr[band_arr == nodata_value] = np.nan

        array[:, :, i - 1] = band_arr

    tile_info['nodata_values'] = no_data

    return array, tile_info

def load_hls_with_request2(year, aoi):
    """
    Load two hls images with the least CC
    Args:
        year: input the year
        aoi: input the aoi - wkt

    Returns:
        the tiles with the least CC
    """

    client_id = CLIENT_ID
    client_secret = CLIENT_SECRET
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    sentinel_token_url = 'https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token'

    oauth.fetch_token(token_url=sentinel_token_url,
                          client_secret=client_secret, include_client_id=True)

    bbox = list(shapely.wkt.loads(aoi).bounds)
    url = 'https://services-uswest2.sentinel-hub.com/api/v1/process'

    request = \
       {
        "input": {
            "bounds": {
              "properties": {
              "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
            },
                "bbox": bbox
            },
            "data": [{
                "type": "hls",
                "dataFilter": {
                    "constellation": "SENTINEL",
                    "mosaickingOrder": "leastCC",
                    "timeRange": {
                        "from": f"{year}-01-01T00:00:00Z",
                        "to": f"{year}-05-31T00:00:00Z"
                    }
                },
            }]
        },
        "output": {
            "height": 512,
            "width": 512,
            "responses": [
             {
                "identifier": "default",
                "format": {
                  "type": "image/tiff"
                }
             },
            {
                "identifier": "userdata",
                "format": {
                    "type": "application/json"
                }
            }
            ]
        },

        "evalscript": """
        //VERSION=3

        function setup() {
            return {
                input: ["Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2"],
                output: {
                    bands: 6
                }
            };
        }
        
        function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
              outputMetadata.userData = { "tiles":  scenes.tiles }
        }

        function evaluatePixel(sample) {
            return [sample.Blue, sample.Green, sample.Red, sample.NIR_Narrow, sample.SWIR1, sample.SWIR2];
        }
        """
        }

    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/x-tar'
    }

    response = oauth.post(url, headers=headers, json=request)

    if response.status_code == 200:
        if response.content:
            try:
                with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                    # Extract userdata.json
                    userdata_file = tar.extractfile('userdata.json')
                    userdata = json.load(userdata_file)

                    # Find the tile with the least cloud coverage
                    selected_tile = min(userdata['tiles'], key=lambda x: x['cloudCoverage'])

                    # Sort tiles by cloudCoverage and select the two with the least coverage
                    sorted_tiles = sorted(userdata['tiles'], key=lambda x: x['cloudCoverage'])
                    best_two_tiles = sorted_tiles[:2]

                    results = []
                    for tile in best_two_tiles:
                        tile_id = tile['tileOriginalId']

                        # Extract TIFF file and read it into a numpy array
                        tiff_file = tar.extractfile(f'default.tif')
                        with MemoryFile(tiff_file.read()) as memfile:
                            with memfile.open() as dataset:
                                tiff_array = dataset.read()  # Read all bands into a numpy array

                        # Transpose the TIFF array from (C, H, W) to (H, W, C)
                        tiff_array_transposed = np.transpose(tiff_array, (1, 2, 0))

                        save_path = './results'

                        # Save the transposed TIFF array to a file
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        file_name = f"hls_{tile_id}_" + str(datetime.now().strftime("%Y-%m-%d_%H%M")) + ".tif"
                        save_file = os.path.join(save_path, file_name)
                        userdata_file_path = os.path.join(save_path, f'hls_{tile_id}_{str(datetime.now().strftime("%Y-%m-%d_%H%M"))}_userdata.json')

                        with rasterio.open(
                            save_file,
                            'w',
                            driver='GTiff',
                            height=tiff_array_transposed.shape[0],
                            width=tiff_array_transposed.shape[1],
                            count=tiff_array_transposed.shape[2],
                            dtype=tiff_array_transposed.dtype,
                            crs='EPSG:4326',
                            transform=dataset.transform
                        ) as dst:
                            for i in range(tiff_array_transposed.shape[2]):
                                dst.write(tiff_array_transposed[:, :, i], i + 1)

                        # Save selected userdata to a JSON file
                        with open(userdata_file_path, 'w') as f:
                            json.dump(userdata, f, indent=4)

                        results.append((tiff_array_transposed, userdata, save_file))

                    print("Selected tiles:", [tile['tileOriginalId'] for tile in best_two_tiles])
                    return results  # Return a list of the two best tiles

            except tarfile.TarError as e:
                print(f"Error extracting tar file: {e}")
                return None, None

        else:
            print("Response content is empty.")
            return None, None
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None, None


def load_hls_with_request(year, aoi):

    client_id = CLIENT_ID
    client_secret = CLIENT_SECRET
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    sentinel_token_url = 'https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token'

    oauth.fetch_token(token_url=sentinel_token_url,
                          client_secret=client_secret, include_client_id=True)

    bbox = list(shapely.wkt.loads(aoi).bounds)
    url = 'https://services-uswest2.sentinel-hub.com/api/v1/process'

    request = \
       {
        "input": {
            "bounds": {
              "properties": {
              "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
            },
                "bbox": bbox
            },
            "data": [{
                "type": "hls",
                "dataFilter": {
                    "constellation": "SENTINEL",
                    "mosaickingOrder": "leastCC",
                    "timeRange": {
                        "from": f"{year}-01-01T00:00:00Z",
                        "to": f"{year}-03-31T00:00:00Z"
                    }
                },
            }]
        },
        "output": {
            "responses": [
             {
                "identifier": "default",
                "format": {
                  "type": "image/tiff"
                }
             },
            {
                "identifier": "userdata",
                "format": {
                    "type": "application/json"
                }
            }
            ]
        },

        "evalscript": """
        //VERSION=3

        function setup() {
            return {
                input: ["Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2"],
                output: {
                    bands: 6
                }
            };
        }
        
        function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
              outputMetadata.userData = { "tiles":  scenes.tiles }
        }

        function evaluatePixel(sample) {
            return [sample.Blue, sample.Green, sample.Red, sample.NIR_Narrow, sample.SWIR1, sample.SWIR2];
        }
        """
        }

    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/x-tar'
    }

    response = oauth.post(url, headers=headers, json=request)

    if response.status_code == 200:
        if response.content:
            try:
                with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                    # Extract userdata.json
                    userdata_file = tar.extractfile('userdata.json')
                    userdata = json.load(userdata_file)

                    # Find the tile with the least cloud coverage
                    selected_tile = min(userdata['tiles'], key=lambda x: x['cloudCoverage'])

                    # Extract the tileOriginalId of the selected tile
                    selected_tile_id = selected_tile['tileOriginalId']
                    print(selected_tile_id)

                    # Extract TIFF file and read it into a numpy array
                    tiff_file = tar.extractfile('default.tif')
                    with MemoryFile(tiff_file.read()) as memfile:
                        with memfile.open() as dataset:
                            tiff_array = dataset.read()  # Read all bands into a numpy array
                            # Print all metadata
                            print("Dataset Metadata:")
                            for key, value in dataset.meta.items():
                                print(f"{key}: {value}")

                # Transpose the TIFF array from (13, 256, 256) to (256, 256, 13)
                tiff_array_transposed = np.transpose(tiff_array, (1, 2, 0))

                save_path = './results'

                # Save the transposed TIFF array to a file
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_name = "hls" +"_"+str(datetime.now().strftime("%Y-%m-%d_%H%M"))+".tif"
                save_file = os.path.join(save_path, file_name)
                userdata_file_path = os.path.join(save_path, f'hls_{str(datetime.now().strftime("%Y-%m-%d_%H%M"))}_userdata.json')

                with rasterio.open(
                    save_file,
                    'w',
                    driver='GTiff',
                    height=tiff_array_transposed.shape[0],
                    width=tiff_array_transposed.shape[1],
                    count=tiff_array_transposed.shape[2],
                    dtype=tiff_array_transposed.dtype,
                    crs='EPSG:4326',  # Adjust CRS if needed
                    transform=dataset.transform
                ) as dst:
                    for i in range(tiff_array_transposed.shape[2]):
                        dst.write(tiff_array_transposed[:, :, i], i + 1)

                # Save selected userdata to a JSON file
                with open(userdata_file_path, 'w') as f:
                    json.dump(userdata, f, indent=4)

                print("Shape of transposed TIFF array:", tiff_array_transposed.shape)
                print(f"Transposed TIFF array saved to {save_file}")

                return tiff_array_transposed, userdata, save_file  # Return both the transposed TIFF array and user data

            except tarfile.TarError as e:
                print(f"Error extracting tar file: {e}")
                return None, None

        else:
            print("Response content is empty.")
            return None, None
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None, None


def get_tile_info(refDataset):
    tile_info = {}
    tile_info['projection'] = refDataset.GetProjection()
    tile_info['geotransform'] = refDataset.GetGeoTransform()
    tile_info['width'] = refDataset.RasterXSize
    tile_info['height'] = refDataset.RasterYSize
    return tile_info


def preprocess_image(image, means, stds):
    # normalize image
    image = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

