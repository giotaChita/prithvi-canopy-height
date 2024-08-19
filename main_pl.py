import comet_ml
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from osgeo import gdal
import click
from utils.config import cache_path, best_model_path_new
from model.Dataset import canopy_height_GEDI, TiledDataset, preprocess_data
from utils.utils import load_hls_with_request2, plot_loaders, load_tif_as_array
import numpy as np
from model.pl_model import CHLighteningModule
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
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
    current_path = os.getcwd()

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
        file_path1 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019045T103129.v2.0_2024-08-13_1307.tif'
        file_path2 = os.getcwd() + '/results/hls_HLS.S30.T32TMS.2019052T102039.v2.0_2024-08-13_1302.tif'

        response1, t1 = load_tif_as_array(file_path1)
        response2, t2 = load_tif_as_array(file_path2)

    print(response1.shape)
    print(response2.shape)


    ds = gdal.Open(file_path1)
    if ds is None:
        raise FileNotFoundError(f"Failed to open HLS image at {file_path1}")


    geotransform = ds.GetGeoTransform()
    input_crs = ds.GetProjection()

    # TODO: find a better way to represent the heigth from gedi shots

    # set to true this the first time if you have not saved the gedi shots to npy file
    download_gedi_shots = False
    if download_gedi_shots:
        canopy_height_labels, _ = canopy_height_GEDI(file_path1, response1)
    else:
        canopy_height_labels = np.load(cache_path)
        canopy_height_labels = torch.tensor(canopy_height_labels, dtype=torch.float32)

    # Scaling images
    hls_data1, response1 = preprocess_data(response1)
    hls_data2, response2 = preprocess_data(response2)

    tile_size = 50
    overlap = tile_size // 2

    dataset1 = TiledDataset(hls_data1, canopy_height_labels, tile_size=tile_size, overlap=overlap)

    # Count non-NaN pixels of canopy height for img1
    valid_pixels, total_pixels = dataset1.count_non_nan_pixels()
    print(f"Number of valid (non-NaN) pixels in the dataset: {valid_pixels}")
    print(f"Total number of pixels in the dataset: {total_pixels}")
    print(f"Percentage of valid pixels: {valid_pixels / total_pixels * 100:.2f}%")

    dataset2 = TiledDataset(hls_data2, canopy_height_labels, tile_size=tile_size, overlap=overlap)

    # Count non-NaN pixels of canopy height for img2
    valid_pixels, total_pixels = dataset2.count_non_nan_pixels()
    print(f"Number of valid (non-NaN) pixels in the dataset: {valid_pixels}")
    print(f"Total number of pixels in the dataset: {total_pixels}")
    print(f"Percentage of valid pixels: {valid_pixels / total_pixels * 100:.2f}%")

    # Combine the two images
    dataset = dataset1 + dataset2

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return total_params, trainable_params, frozen_params

    embed_dims = 768

    # Hyperparameters
    lr = 1e-3
    batch_size = 16 # 8
    epochs = 15 # 50, 100, 250, 500

    hparams = {
            'lr': lr,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'log_interval': 2,
            'step_size': 2,
            'gamma': 0.1
        }

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create a Comet Logger
    comet_logger = CometLogger(
        api_key="YZiwsYqIN87kijoaS5atmnqqz",
        project_name="prithvi-pytorch-lightning",
        workspace="tagio"
    )

    # Initialize the model
    model = CHLighteningModule(tile_size=50, patch_size=16, hparams=hparams)

    total_params, trainable_params, frozen_params = count_parameters(model)

    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print(f"Frozen parameters: {frozen_params / 1e6:.2f} M")

    early_stopping = EarlyStopping('val_loss', patience=7, mode='min', verbose=True)

    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed',
        log_every_n_steps=1,
        logger=comet_logger,
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min'),
                    LearningRateMonitor(logging_interval='epoch'),
                    early_stopping],
    )

    # Learning Rate Finder
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    fig = lr_finder.plot()
    fig.show()

    # Get the suggested learning rate
    new_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {new_lr}")

    # Update hyperparameters with the new learning rate
    hparams['lr'] = new_lr
    model.hparams.update(hparams)

    # Reinitialize the model with the new learning rate
    model = CHLighteningModule(tile_size=50, patch_size=16, hparams=hparams, train_len=len(train_loader))

    # Train the model with the updated learning rate
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the best model
    torch.save(model.best_model_weights, best_model_path_new)

    # Test the model
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()

