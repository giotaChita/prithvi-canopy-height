import rasterio
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_tiff(input_tiff_path, output_tiff_path, dst_crs='EPSG:4326'):
    """
    Reproject a TIFF file to a different CRS.

    Args:
        input_tiff_path (str): Path to the input TIFF file with the original CRS.
        output_tiff_path (str): Path to save the output reprojected TIFF file.
        dst_crs (str): Target coordinate reference system (CRS), default is WGS 84 (EPSG:4326).
    """
    with rasterio.open(input_tiff_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tiff_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def read_tiff(tiff_path):
    """
    Read tiff file and return data
    Args:
        tiff_path: str, path to file

    Returns:
        data: numpy array
        meta data: dict
    """
    with rasterio.open(tiff_path) as img:
        data = img.read(1)
        meta = img.meta
    return data, meta


def resample_image1(data, old_res, new_res):
    """
    Resample image from 10m res to 30m res with bilinear interpolation
    Args:
        data: numpy array
        old_res: float, original resolution
        new_res: float, target resolution

    Returns:
        resample_data: numpy array
    """
    scale_factor = old_res / new_res
    resampled_data = zoom(data, (1/scale_factor, 1/scale_factor), order=1)
    return resampled_data


def save_tiff(data, meta, tiff_path):
    """
    Save the data to a TIFF file with updated metadata.

    Args:
        data (numpy array): The image data.
        meta (dict): Metadata of the original TIFF file.
        tiff_path (str): Path to save the new TIFF file.
    """
    meta.update({
        'dtype': 'float32',
        'count': 1,
        'height': data.shape[0],
        'width': data.shape[1],
        'transform': rasterio.Affine(meta['transform'].a * (meta['width'] / data.shape[1]), 0,
                                     meta['transform'].b, 0,
                                     meta['transform'].e,
                                     meta['transform'].f * (meta['height'] / data.shape[0]))
    })

    with rasterio.open(tiff_path, 'w', **meta) as dst:
        dst.write(data, 1)


def align_images(predicted, ground_truth):
    """
    Crop or pad images to ensure they have the same dimensions.

    Args:
        predicted (numpy array): The predicted image data.
        ground_truth (numpy array): The ground truth image data.

    Returns:
        tuple: Cropped or padded images.
    """
    min_height = min(predicted.shape[0], ground_truth.shape[0])
    min_width = min(predicted.shape[1], ground_truth.shape[1])

    cropped_predicted = predicted[:min_height, :min_width]
    cropped_ground_truth = ground_truth[:min_height, :min_width]

    return cropped_predicted, cropped_ground_truth


def compute_metrics(predicted, ground_truth):
    """
    Compute performance metrics to compare predicted and ground truth data.

    Args:
        predicted (numpy array): The predicted image data.
        ground_truth (numpy array): The ground truth image data.

    Returns:
        tuple: MAE, RMSE, and R2 score.
    """
    # Flatten the arrays
    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()

    # Compute metrics
    mae = mean_absolute_error(ground_truth_flat, predicted_flat)
    rmse = np.sqrt(mean_squared_error(ground_truth_flat, predicted_flat))
    r2 = r2_score(ground_truth_flat, predicted_flat)

    return mae, rmse, r2


def check_statistics(data, name):
    """
    Print basic statistics of the data to help identify extreme values.

    Args:
        data (numpy array): The image data.
        name (str): Name of the dataset for identification.
    """
    print(f"{name} - Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}, Std: {np.std(data)}")


def handle_extreme_values(data):
    """
    Handle extreme values and NaNs/Infs in the data.

    Args:
        data (numpy array): The image data.

    Returns:
        numpy array: Data with extreme values, NaNs, and Infs handled.
    """
    # Replace extreme negative values and Infs with NaNs for handling
    data[data < 0] = np.nan
    data[np.isinf(data)] = np.nan

    # Replace NaNs with the mean of the non-NaN values
    mean_val = np.nanmean(data)
    data[np.isnan(data)] = mean_val

    return data


def normalize_data(data):
    """
    Normalize data to range [0, 1].

    Args:
        data (numpy array): The image data.

    Returns:
        numpy array: Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:  # Avoid division by zero
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = data
    return normalized_data


# Paths to the TIFF files
predicted_tiff_path = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/results/reconstructed_canopy_height_combined_2.tiff'
ground_truth_tiff_path = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/data/GroundTruth_CHE_switzerland_2019_10m.tif'
resampled_ground_truth_tiff_path = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/data/resampled_ground_truth_canopy_height_30m.tiff'
predicted_tiff_path_rep = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/results/reconstructed_canopy_height_combined_30.tiff'

# ground_truth_data, ground_truth_meta = read_tiff(ground_truth_tiff_path)


# Open the ground truth GeoTIFF
with rasterio.open(ground_truth_tiff_path) as src:
    ground_truth = src.read(
        out_shape=(src.count, 512, 512),  # Match the dim with the predicted
        resampling=Resampling.bilinear
    )

# Load the data
with rasterio.open(predicted_tiff_path) as src:
    predicted = src.read(1)

valid_min = 0
# Mask invalid data
ground_truth[ground_truth < valid_min] = np.nan
ground_truth = np.nan_to_num(ground_truth, nan=0)

# Flatten the arrays to 1D
ground_truth_flat = ground_truth.flatten()
predicted_flat = predicted.flatten()

# Check for infinite or NaN values in ground_truth and predicted arrays
print("Ground Truth - Any NaN or Inf values? ", np.any(np.isnan(ground_truth)) or np.any(np.isinf(ground_truth)))
print("Predicted - Any NaN or Inf values? ", np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)))
print("Ground Truth - Min, Max values: ", np.min(ground_truth_flat), np.max(ground_truth_flat))
print("Predicted - Min, Max values: ", np.min(predicted_flat), np.max(predicted_flat))

# Ensure there are no NaN or Inf values
ground_truth_flat = np.nan_to_num(ground_truth_flat, nan=0, posinf=0, neginf=0)
predicted_flat = np.nan_to_num(predicted_flat, nan=0, posinf=0, neginf=0)

print("\n--------METHOD 1---------")

mae = mean_absolute_error(ground_truth_flat, predicted_flat)
print(f"Mean Absolute Error (MAE): {mae}")
mse = mean_squared_error(ground_truth_flat, predicted_flat)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
r2 = r2_score(ground_truth_flat, predicted_flat)
print(f"R² (Coefficient of Determination): {r2}")

# METHOD 2
print("\n--------METHOD 2---------")

# Function to read TIFF data with rasterio
def read_tiff_with_rasterio(tiff_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)  # Read the first band
        meta = src.meta  # Save metadata
    return data, meta

# Read predicted data with rasterio
predicted, predicted_meta = read_tiff_with_rasterio(predicted_tiff_path)

# Read ground truth data with rasterio
ground_truth, ground_truth_meta = read_tiff_with_rasterio(ground_truth_tiff_path)

# Define the target size (512, 512)
target_size = (512, 512)

# Perform resampling using scipy
def resample_image(image, target_size):
    original_shape = image.shape
    zoom_factors = (target_size[0] / original_shape[0], target_size[1] / original_shape[1])
    return zoom(image, zoom_factors, order=1)  # order=1 is bilinear interpolation

# Resample ground truth to match the dimensions of predicted
resampled_ground_truth = resample_image(ground_truth, target_size)

# Handle NaN values by replacing them with zeros
resampled_ground_truth = np.nan_to_num(resampled_ground_truth, nan=0)
predicted = np.nan_to_num(predicted, nan=0)

# Ensure dimensions match
if resampled_ground_truth.shape != predicted.shape:
    raise ValueError(f"Dimension mismatch: resampled_ground_truth shape {resampled_ground_truth.shape}, predicted shape {predicted.shape}")
resampled_ground_truth[resampled_ground_truth < valid_min] = np.nan
resampled_ground_truth = np.nan_to_num(resampled_ground_truth, nan=0)

# Flatten the arrays to 1D
ground_truth_flat = resampled_ground_truth.flatten()
predicted_flat = predicted.flatten()

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(ground_truth_flat, predicted_flat)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Root Mean Square Error (RMSE)
mse = mean_squared_error(ground_truth_flat, predicted_flat)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")

# Calculate R² (Coefficient of Determination)
r2 = r2_score(ground_truth_flat, predicted_flat)
print(f"R² (Coefficient of Determination): {r2}")

# # Original and new resolutions
# original_resolution = 10
# new_resolution = 30
#
# resampled_ground_truth = resample_image(ground_truth_data, original_resolution, new_resolution)

# save_tiff(resampled_ground_truth, ground_truth_meta, resampled_ground_truth_tiff_path)



# ground_truth_data, ground_truth_meta = read_tiff(resampled_ground_truth_tiff_path)

# # Align the images
# aligned_predicted, aligned_ground_truth = align_images(predicted_data, ground_truth_data)
#
# # Calculate metrics
# mae, rmse, r2 = compute_metrics(aligned_predicted, aligned_ground_truth)
#
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R-squared (R2): {r2:.4f}")
#
# # Check statistics for both datasets
# check_statistics(predicted_data, "Predicted Data")
# check_statistics(ground_truth_data, "Ground Truth Data")
#
# # Handle extreme values in both datasets
# handled_predicted = handle_extreme_values(predicted_data)
# handled_ground_truth = handle_extreme_values(ground_truth_data)
#
# aligned_predicted_2, aligned_ground_truth_2 = align_images(handled_predicted, handled_ground_truth)
#
# # Calculate metrics
# mae, rmse, r2 = compute_metrics(aligned_predicted_2, aligned_ground_truth_2)
#
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R-squared (R2): {r2:.4f}")
#
# # Normalize the datasets
# normalized_predicted = normalize_data(handled_predicted)
# normalized_ground_truth = normalize_data(handled_ground_truth)
#
# # Align images
# aligned_predicted, aligned_ground_truth = align_images(normalized_predicted, normalized_ground_truth)
# # Check statistics for both datasets
# check_statistics(handled_predicted, "Predicted Data")
# check_statistics(handled_ground_truth, "Ground Truth Data")
#
# # Calculate metrics
# mae, rmse, r2 = compute_metrics(aligned_predicted, aligned_ground_truth)
#
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R-squared (R2): {r2:.4f}")
