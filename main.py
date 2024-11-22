import click
from utils.config import (cache_path_rh98, device, best_model_path_new, cache_path, cache_path_rh99,
                          cache_path_aoi4_gedi_shots_rh98_28_9, cache_path_aoi5_rh95,
                          pretrained_model_path, best_model_path, cache_path_aoi3_gedi_shots_rh98_size1024, cache_path_aoi4_rh95_new_filter, cache_path_aoi3_rh95_filter3,
                          cache_path_aoi4_rh95_second_filter_waterzero, cache_path_aoi6_rh95,
                          cache_path_aoi4_gedi_shots_rh98, cache_path_aoi3_gedi_shots_rh98, cache_path_aoi4_rh95_filter3, cache_path_aoi4_rh95_old_filter, cache_path_aoi4_rh95_second_filter, cache_path_aoi2_rh95)
from model.Dataset import (canopy_height_GEDI, preprocess_data, AugmentedTiledDataset, TiledDatasetNoPadding)
from utils.utils import (load_hls_with_request2_Sentinel, compute_metrics, plot_loaders, preprocess_image, load_data1,
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

    aoi2 = False

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
        if aoi2:
        # aoi 2, size 512
            file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019045T103129.v2.0_2024-08-13_1307.tif'  # "date": "2019-02-14T10:37:59.630Z", whole img
            file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019052T102039.v2.0_2024-08-13_1302.tif'  # date": "2019-02-21T10:28:02.774Z", with shadow
        else:
            # aoi 2
        #     file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019045T103129.v2.0_2024-08-13_1307.tif'  # "date": "2019-02-14T10:37:59.630Z", whole img
        #     file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019052T102039.v2.0_2024-08-13_1302.tif'

        # # aoi 3, hls size 512
        #     file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2127.tif'
        #     file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2127.tif'

        # aoi 3 , hls size 1024
            file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TLT.2019080T103021.v2.0_2024-09-01_2241.tif'
            file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-01_2241.tif'

        # aoi 3, hls size 1024, landsat & Sentinel
        #     file_path1 = os.getcwd() + '/results/hls_HLS.L30.T32TLT.2019205T101630.v2.0_2024-09-13_1522.tif'
        #     file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMT.2019110T103031.v2.0_2024-09-02_0923.tif'

        # aoi 4, hls from earth search engine
        #     file_path1 = os.getcwd() + '/results/HLS.S30.T32TLS.2019180T103031.v2.0.product1_reprojected.tif'
        #     file_path2 = os.getcwd() + '/results/HLS.S30.T32TLS.2019255T103019.v2.0.product2_reprojected.tif'
        #     file_path3 = os.getcwd() + '/results/HLS.S30.T32TLS.2019230T103031.v2.0.reproject_prod3.tiff'
        #     response3, t3 = load_tif_as_array(file_path3)

        # aoi 5 odzala kokoua
        #     file_path2 = os.getcwd() + '/results/HLS.S30.T33NWB.2022040T091131.v2.0_aoi5.tiff'
        #     file_path1 = os.getcwd() + '/results/HLS.S30.T33NWB.2022360T091401.v2.0_aoi5.tif'
        #
        # # aoi6
        #     file_path1 = os.getcwd() + '/results/HLS.S30.T15RUP.2023138T164849_prod1.tiff'
        #     file_path2 = os.getcwd() + '/results/HLS.S30.T15RUP.2022273T165109_prod2.tiff'

        response1, t1 = load_tif_as_array(file_path1)
        response2, t2 = load_tif_as_array(file_path2)

    print(response1.shape)
    print(response2.shape)
    # print(response3.shape)


    # num of NaN values
    nan_count = np.isnan(response1).sum()
    print(f"The response1 array contains {nan_count} NaN values.")
    nan_count2 = np.isnan(response2).sum()
    print(f"The response2 array contains {nan_count2} NaN values.")
    # nan_count3 = np.isnan(response3).sum()
    # print(f"The response3 array contains {nan_count3} NaN values.")

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
    #     axes[i].imshow(response2[:, :, i], cmap='gray')  # Plot the i-th channel
    #     axes[i].set_title(f'Channel {i + 1}')
    #     axes[i].axis('off')  # Turn off axis labels
    # plt.tight_layout()
    # plt.show()
    # exit()

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
        if aoi2:
            canopy_height_labels = np.load(cache_path_aoi2_rh95)  # cache_path_aoi4_rh95_second_filter_waterzero # cache_path_aoi4_rh95_filter3, cache_path_aoi3_rh95_filter3
            canopy_height_labels = torch.tensor(canopy_height_labels, dtype=torch.float32) #cache_path_aoi4_rh95_second_filter_waterzero
        else:
            canopy_height_labels = np.load(cache_path_aoi5_rh95)  # cache_path_aoi4_rh95_second_filter_waterzero # cache_path_aoi4_rh95_filter3, cache_path_aoi3_rh95_filter3
            canopy_height_labels = torch.tensor(canopy_height_labels,
                                                dtype=torch.float32)  # cache_path_aoi4_rh95_second_filter_waterzero

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

    tile_size = 32  # 128  # 32  # 50
    overlap = 0

    # scale images
    hls_data1, response1 = preprocess_data(response1)
    hls_data2, response2 = preprocess_data(response2)
    print(hls_data2.shape, response1.shape)
    # exit()
    # hls_data3, response3 = preprocess_data(response3)

    load_data_from_disk = True
    folder_output = os.getcwd() + "/data/dataset/"
    if load_data_from_disk:
        if aoi2:
            train_dataset, val_dataset, test_dataset = load_data1(folder_output)
        else:
            train_dataset, val_dataset, test_dataset = load_data(folder_output)

    else:

        # with time dimension
        # dataset = TiledDatasetNoPadding([hls_data1, hls_data2, hls_data3], canopy_height_labels, tile_size=tile_size, overlap=overlap)

        # without time dimension
        dataset1 = TiledDatasetNoPadding([hls_data1], canopy_height_labels, tile_size=tile_size, overlap=overlap)
        dataset2 = TiledDatasetNoPadding([hls_data2], canopy_height_labels, tile_size=tile_size, overlap=overlap)
        # dataset3 = TiledDatasetNoPadding([hls_data3], canopy_height_labels, tile_size, overlap)

        # non-NaN pixels of canopy height for img1
        # valid_pixels, total_pixels = dataset1.count_non_nan_pixels()
        # print(f"Number of valid (non-NaN) pixels in the dataset: {valid_pixels}")
        # print(f"Total number of pixels in the dataset: {total_pixels}")
        # print(f"Percentage of valid pixels: {valid_pixels / total_pixels * 100:.2f}%")

        # Combine images
        dataset = dataset1 + dataset2 #+ dataset3

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

    print(f"Num of training batches: {len(train_loader)}, {len(train_dataset)}")
    print(f"Num of validation batches: {len(val_loader)}, {len(val_dataset)}")
    print(f"Num of test batches: {len(test_loader)}, {len(test_dataset)}")
    print('\n')
    # exit()

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
    w1 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_114922.pth" # tile 80, scientific_balustrade_2350
    w2 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_124440.pth" # regional_persimmon_2366, tile siz 80
    w3 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_150154.pth"  # tile 64, gradual_bracket_6373
    w4 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_134355.pth"  # tile 96, administrative_mall_943
    w5 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_183052.pth"  # unnecessary_wheat_9244, tile 48
    w7 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_014606.pth"  # combined_ox_969 , tile size 80
    w8 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_093159.pth"  # spiritual_gayal_397 , tile 48
    w9 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_121032.pth"  # available_limpet_7986, tile 64
    w10 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_13_131737.pth"  # sophisticated_gazelle_8464, tile 32
    w11 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_14_131003.pth"  # supposed_pavilion_2369, tile 32
    w12 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_14_141730.pth"  # exact_giraffe_9182, tile 96
    w13 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_14_172231.pth"  # unconscious_temple_5609. tile 128
    w14 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_15_141528.pth"  # considerable_templare_45 , tile 80
    w15 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_15_151938.pth"  # bored_booby_1516,tile 48
    w16 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_17_152019.pth"  # efficient_jackfruit_6808, tile 80, epochs early stop, drop out rate 0.2
    w17 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_17_160014.pth"  # marked_pie_9170, tile 80
    w18 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_013039.pth"  # secondary_muskox_3961, tile 32,
    w19 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_18_221511.pth"#tile128, excited_jackal_571
    w20 = "/media/giota/e0c77d18-e407-43fd-ad90-b6dd27f3ac38/Thesis/Model/Model_Code/src/model/save_load_model/best_model_state_2024_11_19_150348.pth" # til 32, added_tropics_4468

    if not train:
        model.load_state_dict(torch.load(w10))
    model.to(device)

    test = False
    if test:
        # Test loop
        test_loop(model, test_loader, device)
        # exit()




if __name__ == "__main__":
    main()
