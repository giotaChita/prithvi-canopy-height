import click
from utils.config import (cache_path_rh98, device, best_model_path_new, cache_path, cache_path_rh99,
                          cache_path_aoi4_gedi_shots_rh98_28_9, cache_path_aoi5_rh95,
                          pretrained_model_path, best_model_path, cache_path_aoi3_gedi_shots_rh98_size1024, cache_path_aoi4_rh95_new_filter, cache_path_aoi3_rh95_filter3, cache_path_aoi4_rh95_second_filter_waterzero,
                          cache_path_aoi4_gedi_shots_rh98, cache_path_aoi3_gedi_shots_rh98, cache_path_aoi4_rh95_filter3, cache_path_aoi4_rh95_old_filter, cache_path_aoi4_rh95_second_filter)
from model.Dataset import (canopy_height_GEDI, preprocess_data, AugmentedTiledDataset, TiledDatasetNoPadding)
from utils.utils import (load_hls_with_request2_Sentinel, compute_metrics, plot_loaders, preprocess_image,
                         compare_canopy_heights, load_tif_as_array, load_hls_with_request_Landsat, load_data, save_data, fill_nan_with_nearest)
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from torch.utils.data import DataLoader, random_split, ConcatDataset

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

    # set true this the first time if you have not downloaded the hls images with leastCC
    request_hls = False

    if request_hls:
        # Load the two tiles with the least cloud coverage from request hls
        tiles = load_hls_with_request2_Sentinel(year, aoi)  # load_hls_with_request2_Sentinel
        if tiles is None or len(tiles) != 2:
            print("Failed to load two tiles or insufficient tiles.")
            return
        (response1, userdata1, file_path1), (response2, userdata2, file_path2) = tiles
        exit()
    else:
        # Load the save tiff hls images

        # aoi 2, size 512
        # file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019045T103129.v2.0_2024-08-13_1307.tif'  # "date": "2019-02-14T10:37:59.630Z", whole img
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019052T102039.v2.0_2024-08-13_1302.tif'  # date": "2019-02-21T10:28:02.774Z", with shadow

        # # aoi 3, hls size 512
        # file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2127.tif'
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2127.tif'

        # aoi 3 , hls size 1024
        # file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2241.tif'
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2241.tif'

        # aoi 3, hls size 1024, landsat & Sentinel
        # file_path1 = os.getcwd() + '/results/hls_HLS.L30.T32TLT.2019205T101630.v2.0_2024-09-13_1522.tif'
        # file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-02_0923.tif'

        # # aoi 4, hls from earth search engine
        file_path1 = os.getcwd() + '/results/HLS.S30.T32TLS.2019180T103031.v2.0.product1_reprojected.tif'
        file_path2 = os.getcwd() + '/results/HLS.S30.T32TLS.2019255T103019.v2.0.product2_reprojected.tif'
        file_path3 = os.getcwd() + '/results/HLS.S30.T32TLS.2019230T103031.v2.0.reproject_prod3.tiff'
        # # aoi 5 odzala kokoua
        # file_path2 = os.getcwd() + '/results/HLS.S30.T33NWB.2022040T091131.v2.0_aoi5.tiff'
        # file_path1 =os.getcwd() + '/results/HLS.S30.T33NWB.2022360T091401.v2.0_aoi5.tif'

        response1, t1 = load_tif_as_array(file_path1)
        response2, t2 = load_tif_as_array(file_path2)
        response3, t3 = load_tif_as_array(file_path3)

    print(response1.shape)
    print(response2.shape)
    print(response3.shape)


    # num of NaN values
    nan_count = np.isnan(response1).sum()
    print(f"The response1 array contains {nan_count} NaN values.")
    nan_count2 = np.isnan(response2).sum()
    print(f"The response2 array contains {nan_count2} NaN values.")
    nan_count3 = np.isnan(response3).sum()
    print(f"The response3 array contains {nan_count3} NaN values.")

    # fill nan values with nearest neighbor method
    response2_filled = np.copy(response2)
    for i in range(response2.shape[2]):
        response2_filled[:, :, i] = fill_nan_with_nearest(response2[:, :, i])

    # no NaN values
    nan_count_after = np.isnan(response2_filled).sum()
    print(f"The response2 array contains {nan_count_after} NaN values after filling.")

    # fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    # axes = axes.ravel()
    # for i in range(6):
    #     axes[i].imshow(response3[:, :, i], cmap='gray')  # Plot the i-th channel
    #     axes[i].set_title(f'Channel {i + 1}')
    #     axes[i].axis('off')  # Turn off axis labels
    # plt.tight_layout()
    # plt.show()

    response2 = response2_filled
    ds = gdal.Open(file_path1)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {file_path1}")

    geotransform = ds.GetGeoTransform()
    input_crs = ds.GetProjection()
    print(geotransform, input_crs)

    # set to true this the first time if you have not saved the gedi shots to npy file
    download_gedi_shots = False

    if download_gedi_shots:
        canopy_height_labels, _ = canopy_height_GEDI(file_path1, response1)
        # canopy_height_labels, _ = canopy_height_GEDI_2(file_path1, response1)
        canopy_height_labels = torch.tensor(canopy_height_labels, dtype=torch.float32)

    else:
        canopy_height_labels = np.load(cache_path_aoi4_rh95_second_filter_waterzero)  # cache_path_aoi4_rh95_filter3, cache_path_aoi3_rh95_filter3
        canopy_height_labels = torch.tensor(canopy_height_labels, dtype=torch.float32)

    # Assuming canopy_height_label is your tensor
    # num_zeros = torch.sum(canopy_height_labels == 0).item()
    # print(f"Number of zero values in Canopy Height label: {num_zeros}")

    # cmap = plt.cm.viridis
    # # cmap.set_bad(color='black')
    # norm = mcolors.Normalize(vmin=np.nanmin(canopy_height_labels), vmax=np.nanmax(canopy_height_labels))
    # plt.figure(figsize=(6, 6))
    # plt.imshow(canopy_height_labels, cmap=cmap, norm=norm, interpolation='none')
    # plt.colorbar(label='Canopy Height (m)')
    # plt.title('Canopy Height Labels')
    # plt.show()

    compare = False

    if compare:
        compare_canopy_heights(cache_path_rh98, cache_path_rh99, cache_path)

    tile_size = 32  # 64  # 32  # 50
    overlap = 0

    # scale images
    hls_data1, response1 = preprocess_data(response1)
    hls_data2, response2 = preprocess_data(response2)
    hls_data3, response3 = preprocess_data(response3)

    load_data_from_disk = True
    folder_output = os.getcwd() + "/data/dataset/"
    if load_data_from_disk:
        train_dataset, val_dataset, test_dataset = load_data(folder_output)
    else:

        # with time dimension
        # dataset = TiledDatasetNoPadding([hls_data1, hls_data2, hls_data3], canopy_height_labels, tile_size=tile_size, overlap=overlap)

        # without time dimension
        dataset1 = TiledDatasetNoPadding([hls_data1], canopy_height_labels, tile_size=tile_size, overlap=overlap)
        dataset2 = TiledDatasetNoPadding([hls_data2], canopy_height_labels, tile_size=tile_size, overlap=overlap)
        dataset3 = TiledDatasetNoPadding([hls_data3], canopy_height_labels, tile_size, overlap)

        # non-NaN pixels of canopy height for img1
        valid_pixels, total_pixels = dataset1.count_non_nan_pixels()
        print(f"Number of valid (non-NaN) pixels in the dataset: {valid_pixels}")
        print(f"Total number of pixels in the dataset: {total_pixels}")
        print(f"Percentage of valid pixels: {valid_pixels / total_pixels * 100:.2f}%")

        # Combine images
        dataset = dataset1 + dataset2 + dataset3

        train_size = int(0.7 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        print('before augmentation ', len(train_dataset))

        # Apply Augmentation

        train_dataset_augmented = AugmentedTiledDataset(train_dataset)
        train_dataset = ConcatDataset([train_dataset, train_dataset_augmented])
        print('after augmentation ', len(train_dataset))

        # save data:
        save_data(train_dataset, val_dataset, test_dataset, folder_output)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # plot_loaders(train_loader, val_loader, test_loader)
    num_frames = 0
    for batch_x, batch_y in val_loader:
        num_frames = batch_x.size(2)
        print(f'Validation - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break
    print(num_frames)
    for batch_x, batch_y in test_loader:

        print(f'Test - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break

    for batch_x, batch_y in train_loader:
        print(f'Train - batch_x: {batch_x.shape}, batch_y: {batch_y.shape}')
        break

    print(f"Num of training batches: {len(train_loader)}")
    print(f"Num of validation batches: {len(val_loader)}")
    print(f"Num of test batches: {len(test_loader)}")
    print('\n')

    # the model
    patch_size = 16

    model = MaskedAutoencoderViT(img_size=tile_size, patch_size=patch_size,
                                 num_frames=num_frames, tubelet_size=1, in_chans=6, embed_dim=768, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
    model.to(device)
    # print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, Trainable layers: {trainable_layers}")

    train = False  # Set to true to rerun the training, validation, test process

    if train:
        # Train and Validation Loop
        train_val_loop(model, device, batch_size, patch_size, tile_size, train_loader, val_loader, overlap)

    else:
        # Load pretrained weights
        pretrained_weights = torch.load(pretrained_model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_weights.items() if
                           k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    # Load best model

    # w = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_09_10_010030.pth"
    # w = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_09_13_124721.pth"
    # w = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_09_01_231922.pth'
    # w = '/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_09_13_140504.pth'
    w = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_05_162728.pth"
    w2 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_03_203911.pth"
    w3 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_10_180050.pth"
    w4 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_11_005412.pth" # TILE 32, BATCH 16
    w5 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_10_222101.pth"
    w6 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_11_235725.pth"
    w7 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_12_004719.pth" # tile size 80,batch 8
    w8 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_12_213940.pth" # # tile size 80,batch 8
    w9 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_014606.pth" # best mae 5.6, tile size 80, batch 8
    w10 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_093159.pth" # best mae for tile size 48, batch 16, mae test 5.1
    w11 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_131737.pth" # tile size 32, mae test: 5.1
    if not train:
        model.load_state_dict(torch.load(w11))
    model.to(device)

    test = True
    if test:
        # Test loop
        test_loop(model, test_loader, device)
        # exit()

    ### Method 1 ###
    # exit()
    def reconstruct_image_merge(tiles, tile_size, overlap, img_size):
        reconstructed_img = np.zeros(img_size, dtype=np.float32)
        count_matrix = np.zeros(img_size, dtype=np.float32)

        grid_size_y = (img_size[0] - overlap) // (tile_size - overlap)
        grid_size_x = (img_size[1] - overlap) // (tile_size - overlap)

        idx = 0
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                start_row = i * (tile_size - overlap)
                start_col = j * (tile_size - overlap)
                end_row = start_row + tile_size
                end_col = start_col + tile_size
                reconstructed_img[start_row:end_row, start_col:end_col] += tiles[idx]
                count_matrix[start_row:end_row, start_col:end_col] += 1
                idx += 1

        overlap_mask = count_matrix > 1
        reconstructed_img[overlap_mask] /= count_matrix[overlap_mask]

        return reconstructed_img

    def predict_and_reconstruct(image_data, model, tile_size, overlap, canopy_height_labels):
        dataset = TiledDatasetNoPadding(image_data, canopy_height_labels, tile_size=tile_size, overlap=overlap)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions_list = []

        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                pred = model(images)
                predictions_list.append(pred.detach().cpu().numpy())

        predictions = np.concatenate(predictions_list, axis=0).squeeze()
        img_height, img_width = image_data[0].shape[1], image_data[0].shape[2]
        return reconstruct_image_merge(predictions, tile_size, overlap, (img_height, img_width))

    # Predict and reconstruct for both images
    # reconstructed_predictions1 = predict_and_reconstruct(hls_data1, model, tile_size, overlap, canopy_height_labels)
    reconstructed_predictions2 = predict_and_reconstruct([hls_data1], model, tile_size, overlap, canopy_height_labels)

    # Combine predictions from both images
    # combined_predictions = np.minimum(reconstructed_predictions1, reconstructed_predictions2)

    # Save the combined predictions
    transform = Affine.from_gdal(*geotransform)
    output_tiff_path = os.getcwd() + '/results/best_model_aoi4_13_11_filter_4_tile32.tiff'
    out_meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': hls_data1.shape[2],
        'height': hls_data1.shape[1],
        'count': 1,
        'crs': input_crs,
        'transform': transform
    }

    with rasterio.open(output_tiff_path, 'w', **out_meta) as dst:
        dst.write(reconstructed_predictions2, 1)
    exit()

    ### Method 2 ###

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
    image_test = np.concatenate(image_list, axis=0).squeeze()
    test_mae, test_rmse, test_r2 = compute_metrics(predictions_test, targets_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test R-squared: {test_r2:.4f}")

    # # take the non nan values to compare:
    # mask = ~np.isnan(targets_test)
    # target_masked = targets_test[mask]
    # predicted_masked = predictions_test[mask]
    #
    # predictions_test_flat_masked = predicted_masked.flatten()
    # targets_test_flat_masked = target_masked.flatten()
    #
    # # Plot both histograms in the same plot with transparency
    # plt.figure(figsize=(12, 7))
    #
    # # Plot predicted data
    # sns.histplot(predictions_test_flat_masked, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.4,
    #              label='Predicted CH')
    #
    # # Plot ground truth (targets) data
    # sns.histplot(targets_test_flat_masked, bins=30, kde=True, color='darkorange', edgecolor='black', alpha=0.4,
    #              label='Ground Truth- CH')
    #
    # # Add labels, title, and legend
    # plt.xlabel('Canopy height')
    # plt.ylabel('Frequency')
    # plt.title('Predicted vs Ground Truth Canopy Height')
    # plt.legend()
    #
    # plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()
    #
    # # Improved hexbin plot
    # plt.figure(figsize=(8, 6))
    #
    # # Increase gridsize and use a logarithmic color scale to show data density
    # hb = plt.hexbin(targets_test_flat_masked, predictions_test_flat_masked, gridsize=50, cmap='inferno', mincnt=1, bins='log', alpha=0.8)
    #
    # # Add a colorbar with a label
    # cb = plt.colorbar(hb)
    # cb.set_label('log10(count)')
    #
    # # Add a regression line to show the trend
    # sns.regplot(x=targets_test_flat_masked, y=predictions_test_flat_masked, scatter=False, color='blue',
    #             line_kws={"linewidth": 1.5, "alpha": 0.7})
    #
    # # Add labels and title
    # plt.xlabel('Target Canopy Height (Valid CH values)')
    # plt.ylabel('Predicted Canopy Height (Valid CH values)')
    # plt.title('Target vs Predicted Canopy Heights (Valid Data)')
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    # targets_test_torch = torch.tensor(targets_test, dtype=torch.float32)

    # valid_pixels = torch.sum(~torch.isnan(targets_test_torch))
    # total_pixels = targets_test_torch.numel()
    # print( valid_pixels.item(), total_pixels)
    # Loop through every 6th index in the specified range
    for i in range(1, 400, 1):
        # Initialize a single figure with a wide aspect ratio for 4 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Predicted Canopy Heights - Time 0
        im0 = axs[0].imshow(predictions_test[i, :, :], cmap='viridis', vmin=0, vmax=50)
        axs[0].set_title('Predicted Canopy Heights - time 0')
        fig.colorbar(im0, ax=axs[0])

        # Predicted Canopy Heights - Time 1
        # im1 = axs[1].imshow(predictions_test[i, 1, :, :], cmap='viridis', vmin=0, vmax=50)
        # axs[1].set_title('Predicted Canopy Heights - time 1')
        # fig.colorbar(im1, ax=axs[1])

        # Ground Truth Canopy Heights from GEDI
        im2 = axs[1].imshow(targets_test[i, :, :], cmap='viridis', vmin=0, vmax=50)
        axs[1].set_title('Ground Truth Canopy Heights from GEDI')
        fig.colorbar(im2, ax=axs[1])

        # Image Tile (HLS)
        image_tile = image_test[i, 0, :, :]  # Adjusted indexing based on your initial structure
        im3 = axs[2].imshow(image_tile, cmap='gray')
        axs[2].set_title('Image HLS')
        fig.colorbar(im3, ax=axs[2])

        # Adjust layout for a better fit
        plt.tight_layout()
        plt.show()
    # for i in range(1, 400, 6):
    #     # Plotting final results
    #     plt.figure(figsize=(16, 6))
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(predictions_test[i, 0, :, :], cmap='viridis', vmin=0, vmax=50)
    #     plt.title('Predicted Canopy Heights - time 0')
    #     plt.colorbar()
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(predictions_test[i, 1, :, :], cmap='viridis', vmin=0, vmax=50)
    #     plt.title('Predicted Canopy Heights - time 1')
    #     plt.colorbar()
    #
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(targets_test[i, 0, :, :], cmap='viridis', vmin=0, vmax=50)
    #     plt.title('Ground Truth Canopy Heights from GEDI')
    #     plt.colorbar()
    #     #
    #     image_tile = image_list[i]
    #     image_tile = image_tile[0,1,0,:,:]
    #     plt.subplot(1, 4, 4)
    #     plt.imshow(image_tile, cmap='gray')
    #     plt.title('Image HLS')
    #     plt.colorbar()
    #
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    main()
