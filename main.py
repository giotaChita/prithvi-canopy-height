import click
from utils.config import cache_path_rh98, device, best_model_path_new, cache_path, cache_path_rh99,pretrained_model_path, best_model_path, cache_path_aoi3_gedi_shots_rh98_size1024
from model.Dataset import canopy_height_GEDI, TiledDataset, preprocess_data
from utils.utils import load_hls_with_request2, compute_metrics, plot_loaders, preprocess_image, \
    load_model_with_checkpoint, load_tif_as_array
import numpy as np
from torch.utils.data import DataLoader, random_split
from prithvi.Prithvi import MaskedAutoencoderViT
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.train_test_loop import train_val_loop, test_loop
import rasterio
from rasterio.transform import from_origin, Affine
from osgeo import gdal
import os


@click.command(
    short_help="Canopy Height Estimation",
    help="Canopy height estimation based on hls",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--year",
    "-y",
    "year",
    help="Year selected",
    required=True,
    default=None
)
@click.option(
    "--aoi",
    "-a",
    "aoi",
    help="Area of interest in Well-known Text (WKT)",
    required=True,
    default=None,
)
@click.pass_context
def main(ctx, year, aoi):
    print("Lets start the test")

    print("year = ", year)

    # set true this the first time if you have not downloaded the hls images with leastCC
    request_hls = False

    if request_hls:
        # Load the two tiles with the least cloud coverage from request hls
        tiles = load_hls_with_request2(year, aoi)
        if tiles is None or len(tiles) != 2:
            print("Failed to load two tiles or insufficient tiles.")
            return
        (response1, userdata1, file_path1), (response2, userdata2, file_path2) = tiles
    else:
        # Load the save tiff hls images

        ## aoi 2
        # file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019045T103129.v2.0_2024-08-13_1307.tif'
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019052T102039.v2.0_2024-08-13_1302.tif'

        # # aoi 3, hls size 512
        # file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2127.tif'
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2127.tif'
        #
        # response1, t1 = load_tif_as_array(file_path1)
        # response2, t2 = load_tif_as_array(file_path2)

        # aoi 3 , hls size 1024
        file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2241.tif'
        file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2241.tif'

        response1, t1 = load_tif_as_array(file_path1)
        response2, t2 = load_tif_as_array(file_path2)

    print(response1.shape)
    print(response2.shape)

    ds = gdal.Open(file_path1)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {file_path1}")

    geotransform = ds.GetGeoTransform()
    input_crs = ds.GetProjection()

    # TODO: gedi shots - circular Max Pooling
    # TODO: Visualize the tiles corresponding to labels - ground truth - predicted 
    # TODO: Try tiles bigger size

    # set to true this the first time if you have not saved the gedi shots to npy file
    download_gedi_shots = False
    if download_gedi_shots:
        canopy_height_labels, _ = canopy_height_GEDI(file_path1, response1)
        canopy_height_labels = torch.tensor(canopy_height_labels, dtype=torch.float32)
    else:
        canopy_height_labels_rh98 = np.load(cache_path_rh98)
        canopy_height_labels_rh99 = np.load(cache_path_rh99)
        canopy_height_labels_rh100 = np.load(cache_path)

        canopy_height_labels_rh98_hls1024 = np.load(cache_path_aoi3_gedi_shots_rh98_size1024)
        canopy_height_labels = torch.tensor(canopy_height_labels_rh98_hls1024, dtype=torch.float32)

    compare = False

    if compare:
        canopy_height_labels_rh98 = np.load(cache_path_rh98)
        canopy_height_labels_rh99 = np.load(cache_path_rh99)
        canopy_height_labels_rh100 = np.load(cache_path)
        # Compare rh100, rh99 rh98
        canopy_height_flat = canopy_height_labels_rh100.flatten()
        canopy_height_rh99_flat = canopy_height_labels_rh99.flatten()
        canopy_height_rh98_flat = canopy_height_labels_rh98.flatten()

        # Remove NaN values
        canopy_height_flat = canopy_height_flat[~np.isnan(canopy_height_flat)]
        canopy_height_rh98_flat = canopy_height_rh98_flat[~np.isnan(canopy_height_rh98_flat)]
        canopy_height_rh99_flat = canopy_height_rh99_flat[~np.isnan(canopy_height_rh99_flat)]

        # Plot histograms
        plt.figure(figsize=(10, 6))

        plt.hist(canopy_height_flat, bins=50, alpha=0.5, label='Canopy Height RH100', color='orange')
        plt.hist(canopy_height_rh99_flat, bins=50, alpha=0.5, label='Canopy Height RH99', color='blue')
        plt.hist(canopy_height_rh98_flat, bins=50, alpha=0.5, label='Canopy Height RH98', color='red')

        plt.title('Comparison of Canopy Heights')
        plt.xlabel('Canopy Height (m)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Scaling images
    hls_data1, response1 = preprocess_data(response1)
    hls_data2, response2 = preprocess_data(response2)

    tile_size = 100
    # overlap = tile_size // 2
    overlap = 0
    dataset1 = TiledDataset(hls_data1, canopy_height_labels, tile_size=tile_size, overlap=overlap)

    # Count non-NaN pixels of canopy height for img1
    valid_pixels, total_pixels = dataset1.count_non_nan_pixels()
    print(f"Number of valid (non-NaN) pixels in the dataset: {valid_pixels}")
    print(f"Total number of pixels in the dataset: {total_pixels}")
    print(f"Percentage of valid pixels: {valid_pixels / total_pixels * 100:.2f}%")

    dataset2 = TiledDataset(hls_data2, canopy_height_labels, tile_size=tile_size, overlap=overlap)

    # Combine the two images
    dataset = dataset1 + dataset2

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # plot_loaders(train_loader, val_loader, test_loader)

    for batch_x, batch_y in val_loader:
        print(f'Validation - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break

    for batch_x, batch_y in test_loader:
        print(f'Test - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break

    for batch_x, batch_y in train_loader:
        print(f'Train - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break

    # Print dataset and DataLoader details
    print(f"Num of training batches: {len(train_dataset)}")
    print(f"Num of validation batches: {len(val_dataset)}")
    print(f"Num of test batches: {len(test_dataset)}")
    print('\n')

    # the model
    patch_size = 16

    model = MaskedAutoencoderViT(img_size=tile_size, patch_size=patch_size,
                                 num_frames=1, tubelet_size=1, in_chans=6, embed_dim=768, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

    # print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, Trainable layers: {trainable_layers}")

    train = True  # Set to true to rerun the training, validation, test process

    if train:
        # Train and Validation Loop
        train_val_loop(model, device, batch_size, patch_size, tile_size, train_loader, val_loader)

    # Load best model
    weight = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_08_31_163305.pth'
    weight2 = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_09_01_124424.pth'
    # model.load_state_dict(torch.load(weight))
    # exit()
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    test = True
    if test:
        # Test loop
        test_loop(model, test_loader, device)

    ### Method 1 ###

    # def reconstruct_image_merge(tiles, tile_size, overlap, img_size):
    #     reconstructed_img = np.zeros(img_size, dtype=np.float32)
    #     count_matrix = np.zeros(img_size, dtype=np.float32)
    #
    #     grid_size_y = (img_size[0] - overlap) // (tile_size - overlap)
    #     grid_size_x = (img_size[1] - overlap) // (tile_size - overlap)
    #
    #     idx = 0
    #     for i in range(grid_size_y):
    #         for j in range(grid_size_x):
    #             start_row = i * (tile_size - overlap)
    #             start_col = j * (tile_size - overlap)
    #             end_row = start_row + tile_size
    #             end_col = start_col + tile_size
    #             reconstructed_img[start_row:end_row, start_col:end_col] += tiles[idx]
    #             count_matrix[start_row:end_row, start_col:end_col] += 1
    #             idx += 1
    #
    #     overlap_mask = count_matrix > 1
    #     reconstructed_img[overlap_mask] /= count_matrix[overlap_mask]
    #
    #     return reconstructed_img
    #
    # def predict_and_reconstruct(image_data, model, tile_size, overlap, canopy_height_labels):
    #     dataset = TiledDataset(image_data, canopy_height_labels, tile_size=tile_size, overlap=0)
    #     loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     predictions_list = []
    #
    #     model.eval()
    #     with torch.no_grad():
    #         for images, _ in loader:
    #             images = images.to(device)
    #             pred = model(images)
    #             predictions_list.append(pred.detach().cpu().numpy())
    #
    #     predictions = np.concatenate(predictions_list, axis=0).squeeze()
    #     img_height, img_width = image_data.shape[1], image_data.shape[2]
    #     return reconstruct_image_merge(predictions, tile_size, overlap, (img_height, img_width))
    #
    # # Predict and reconstruct for both images
    # reconstructed_predictions1 = predict_and_reconstruct(hls_data1, model, tile_size, overlap, canopy_height_labels)
    # reconstructed_predictions2 = predict_and_reconstruct(hls_data2, model, tile_size, overlap, canopy_height_labels)
    #
    # # Combine predictions from both images
    # combined_predictions = np.minimum(reconstructed_predictions1, reconstructed_predictions2)
    #
    # # Save the combined predictions
    # transform = Affine.from_gdal(*geotransform)
    # output_tiff_path = os.getcwd() + '/results/reconstructed_canopy_height_combined_3.tiff'
    # out_meta = {
    #     'driver': 'GTiff',
    #     'dtype': 'float32',
    #     'nodata': None,
    #     'width': hls_data1.shape[2],
    #     'height': hls_data1.shape[1],
    #     'count': 1,
    #     'crs': input_crs,
    #     'transform': transform
    # }
    #
    # with rasterio.open(output_tiff_path, 'w', **out_meta) as dst:
    #     dst.write(combined_predictions, 1)

    ### Method 2 ###

    # Recreate dataset to plot for the whole img -> zero overlap
    # dataset = TiledDataset(hls_data1, canopy_height_labels, tile_size=50, overlap=0)
    # test_size = len(dataset)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [0, 0, test_size])
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)0

    dataset = TiledDataset(hls_data1, canopy_height_labels, tile_size=tile_size, overlap=overlap)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Test loop - img

    model.eval()
    test_loss = 0.0
    predictions_list = []
    targets_list = []
    image_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = model.forward_loss(pred, labels)
            test_loss += loss.item()
            predictions_list.append(pred.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())
            image_list.append(images.detach().cpu().numpy())

    test_loss /= len(test_loader)

    # Compute metrics for test set
    predictions_test = np.concatenate(predictions_list, axis=0).squeeze()
    targets_test = np.concatenate(targets_list, axis=0).squeeze()
    test_mae, test_rmse, test_r2 = compute_metrics(predictions_test, targets_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test R-squared: {test_r2:.4f}")

    # targets_test_torch = torch.tensor(targets_test, dtype=torch.float32)

    # valid_pixels = torch.sum(~torch.isnan(targets_test_torch))
    # total_pixels = targets_test_torch.numel()
    # print( valid_pixels.item(), total_pixels)

    for i in range(1, 400, 40):
        # Plotting final results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(predictions_test[i, :, :], cmap='viridis', vmin=0, vmax=50)
        plt.title('Predicted Canopy Heights')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(targets_test[i, :, :], cmap='viridis', vmin=0, vmax=50)
        plt.title('Ground Truth Canopy Heights from GEDI')
        plt.colorbar()
        #
        image_tile = image_list[i]
        image_tile = image_tile[0,1,0,:,:]
        plt.subplot(1, 3, 3)
        plt.imshow(image_tile, cmap='gray')
        plt.title('Image HLS')
        plt.colorbar()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
