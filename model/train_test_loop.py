from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch.nn as nn
from torch_lr_finder import LRFinder
import optuna
import torch
from utils.config import pretrained_model_path, best_model_path_new
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from copy import deepcopy
from utils.utils import compute_metrics
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader


def train_val_loop(model, device, batch_size, patch_size, tile_size, train_loader, val_loader, overlap):
        # Load the pretrained weights
        pretrained_weights = torch.load(pretrained_model_path)
        # model.load_state_dict(pretrained_weights, strict=False)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_weights.items() if
                           k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        # Learning Rate
        lr = 0.001

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        for param in model.canopy_height_head.parameters():
            param.requires_grad = True

        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Number of not trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")

        # num_param = {sum(p.numel() for p in model.parameters() if p.requires_grad)}
        # Comet
        experiment = Experiment(
            api_key="YZiwsYqIN87kijoaS5atmnqqz",
            project_name=  "aoi4-hls-gedi", #"prithvi-original-aoi3", # "aoi5-prithvi",  # "prithvi-original-aoi3"
            workspace="tagio"

        )

        find_best_lr = False

        if find_best_lr:
            optimizer = optim.AdamW(model.canopy_height_head.parameters(), lr=lr, eps=1e-8)
            lr_finder = LRFinder(model, optimizer, model.forward_loss, device="cuda")
            lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
            if any(np.isnan(loss) for loss in lr_finder.history['loss']):
                print("Warning: NaN loss detected during learning rate search.")
            if any(loss <= 0 for loss in lr_finder.history['loss']):
                print("Warning: Non-positive loss values detected.")
            lr_finder.plot()

            for lr, loss in zip(lr_finder.history['lr'], lr_finder.history['loss']):
                experiment.log_metric("learning_rate", lr)
                experiment.log_metric("loss", loss)
            exit()

        num_epochs = 250
        steps_per_epoch = len(train_loader)

        lr = 0.001
        # lr = 5.12E-01

        optimizer = optim.AdamW(model.canopy_height_head.parameters(), lr=lr, eps=1e-8, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) # T_max=100)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5)

        scaler = torch.cuda.amp.GradScaler()

        # lists to store loss values
        train_losses = []
        val_losses = []
        mae_list_val = []
        mae_list_train = []

        # variables to track the best model
        best_val_mae = float('inf')

        # Log hyperparameters to Comet ML
        experiment.log_parameters({
            'Model head': 'Prithvi new Layer',
            'Lr finder': False,
            'batch_size': batch_size,
            'learning_rate': lr,
            'num_epochs': num_epochs,
            'tile_size': tile_size,
            'patch_size': patch_size,
            'rh metric:': 'rh95',
            'aoi': 'aoi4',
            'overlap': overlap,
            'Augmentation': True,
            'Weights_init': True,
            'Scheduler': True
        })

        # Training loop
        for epoch in range(num_epochs):

            model.train()
            running_loss = 0.0
            count = 0
            predictions_list_train = []
            targets_list_train = []

            for inputs, labels in train_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = ~torch.isnan(labels)

                if torch.isnan(inputs).all() or mask.sum() == 0:
                    print("skip")
                    continue
                if torch.isnan(inputs).any():
                    print("skip")
                    continue
                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    pred = model(inputs)
                    loss = model.forward_loss(pred, labels)

                    predictions_list_train.append(pred.detach().cpu().numpy())
                    targets_list_train.append(labels.detach().cpu().numpy())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                experiment.log_metric('grad_norm', grad_norm.item(), step=epoch * steps_per_epoch + count)
                current_lr = scheduler.get_last_lr()[0]
                experiment.log_metric('learning_rate', current_lr, step=epoch * steps_per_epoch + count)

                running_loss += loss.item()
                count += 1

            predictions_train = np.concatenate(predictions_list_train, axis=0)
            targets_train = np.concatenate(targets_list_train, axis=0)

            # Reshape to (batch_size * H * W,)
            predictions_train = predictions_train.reshape(-1)
            targets_train = targets_train.reshape(-1)
            mask_train = ~np.isnan(targets_train)

            # Apply filter mask
            filtered_predictions_train = predictions_train[mask_train]
            filtered_targets_train = targets_train[mask_train]

            num_batches = len(predictions_list_train)
            train_loss = running_loss / num_batches
            # train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            if np.all(np.isnan(filtered_targets_train)) > 0 or np.all(np.isnan(filtered_predictions_train)):
                mae_train = float('nan')
            else:
                mae_train = mean_absolute_error(filtered_targets_train, filtered_predictions_train)
                mae_list_train.append(mae_train)

            experiment.log_metric('train_loss', train_loss, step=epoch + 1)
            experiment.log_metric('train_mae', mae_train, step=epoch + 1)

            # Validation loop
            model.eval()
            val_loss = 0.0
            predictions_list_val = []
            targets_list_val = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    mask = ~torch.isnan(labels)
                    if mask.sum() == 0:
                        continue  # Skip batch

                    pred = model(images)
                    loss = model.forward_loss(pred, labels)
                    val_loss += loss.item()

                    predictions_list_val.append(pred.detach().cpu().numpy())
                    targets_list_val.append(labels.detach().cpu().numpy())

            # some batches are skipped
            num_batches = len(predictions_list_val)
            val_loss /= num_batches
            # val_loss /= len(val_loader)
            val_losses.append(val_loss)

            predictions_val = np.concatenate(predictions_list_val, axis=0)
            targets_val = np.concatenate(targets_list_val, axis=0)
            # Reshape to (batch_size * H * W,)
            predictions_val = predictions_val.reshape(-1)
            targets_val = targets_val.reshape(-1)
            mask_val = ~np.isnan(targets_val)
            # Apply filter mask
            filtered_predictions_val = predictions_val[mask_val]
            filtered_targets_val = targets_val[mask_val]

            if np.all(np.isnan(filtered_targets_val)) or np.all(np.isnan(filtered_predictions_val)):
                mae_val = float('nan')
            else:
                mae_val = mean_absolute_error(filtered_targets_val, filtered_predictions_val)
                mae_list_val.append(mae_val)

            # scheduler.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mae val : {mae_val:.4f}")

            # Log metrics to Comet ML
            # experiment.log_metric('train_loss', train_loss, step=epoch + 1)
            experiment.log_metric('val_loss', val_loss, step=epoch + 1)
            # experiment.log_metric('train_mae', mae_train, step=epoch + 1)
            experiment.log_metric('val_mae', mae_val, step=epoch + 1)

            scheduler.step(mae_val)

            if mae_val < best_val_mae:
                best_val_mae = mae_val
                best_model_state = deepcopy(model.state_dict())

        # Save the best model
        torch.save(best_model_state, best_model_path_new)
        # Load best state
        model.load_state_dict(best_model_state)

        # Plot the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), mae_list_train, label='Train MAE')
        plt.plot(range(1, num_epochs + 1), mae_list_val, label='VAL MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mae')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        plt.show()

        experiment.end()

def test_loop(model, test_loader, device):
    # Test loop
    model.eval()
    test_loss = 0.0
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = model.forward_loss(pred, labels)
            test_loss += loss.item()

            predictions_list.append(pred.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())

    test_loss /= len(test_loader)

    # Compute metrics for test set
    predictions_test = np.concatenate(predictions_list, axis=0).squeeze()
    targets_test = np.concatenate(targets_list, axis=0).squeeze()
    test_mae, test_rmse, test_r2 = compute_metrics(predictions_test, targets_test)
    print("\nTest Set Evaluation Metrics")
    print("=" * 30)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test MAE      : {test_mae:.4f}")
    print(f"Test RMSE     : {test_rmse:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print("=" * 30)

    # # Create a loop to plot each tile one by one
    # for i in range(len(predictions_test)):
    #     plt.figure(figsize=(12, 6))
    #
    #     # Predicted height map
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(predictions_test[i], cmap='viridis', vmin=0, vmax=50)  # Using original values with specified vmin/vmax
    #     plt.colorbar(label='Height (m)')  # Add color bar for predicted height
    #     plt.title(f'Predicted Canopy Height: Tile {i + 1}')
    #     plt.axis('off')  # Turn off axes
    #
    #     # Get the indices of GEDI shots in the ground truth
    #     gedi_shots_indices = np.argwhere(~np.isnan(targets_test[i]))  # Get the positions of valid GEDI shots
    #     predicted_values = []
    #     ground_truth_values = []
    #
    #     # Overlay GEDI shots and get corresponding values
    #     for shot in gedi_shots_indices:
    #         row, col = shot  # Extract row and column from the indices
    #
    #         # Highlight the pixel by creating a border around it (only in the predicted height map)
    #         plt.gca().add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='cyan', linewidth=2, fill=False))  # Highlight border
    #
    #         # Retrieve the predicted and ground truth values
    #         predicted_value = predictions_test[i][row, col]
    #         ground_truth_value = targets_test[i][row, col]
    #
    #         predicted_values.append(predicted_value)
    #         ground_truth_values.append(ground_truth_value)
    #
    #     # Ground truth height map
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(targets_test[i], cmap='viridis', vmin=0, vmax=50)  # Using original values with specified vmin/vmax
    #     plt.colorbar(label='Height (m)')  # Add color bar for ground truth height
    #     plt.title(f'Ground Truth GEDI: Tile {i + 1}')
    #     plt.axis('off')  # Turn off axes
    #
    #     # Overlay GEDI shots on ground truth map (no borders)
    #     for shot in gedi_shots_indices:
    #         row, col = shot  # Extract row and column from the indices
    #         # No border for ground truth
    #
    #     plt.tight_layout()
    #     plt.show()  # Display the current tile plots
    #
    #     # Compare predicted vs ground truth values at GEDI shot locations
    #     print(f"Tile {i + 1} GEDI Shot Values Comparison:")
    #     for idx, shot in enumerate(gedi_shots_indices):
    #         row, col = shot  # Extract row and column from the indices
    #         print(
    #             f"  GEDI Shot at {shot}: Predicted Height = {predicted_values[idx]:.2f}, Ground Truth Height = {ground_truth_values[idx]:.2f}")

    metrics_df = pd.DataFrame({
            'Metric': ['Test Loss', 'Test MAE', 'Test RMSE', 'Test R-squared'],
            'Value': [test_loss, test_mae, test_rmse, test_r2]
        })

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Metric', y='Value', data=metrics_df, hue='Metric', palette='viridis', dodge=False, legend=False)
    plt.title('Test Set Evaluation Metrics', fontsize=16)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Compare the non NaN values
    mask = ~np.isnan(targets_test)
    target_masked = targets_test[mask]
    predicted_masked = predictions_test[mask]

    predictions_test_flat_masked = predicted_masked.flatten()
    targets_test_flat_masked = target_masked.flatten()

    # Plot histograms
    plt.figure(figsize=(12, 7))
    # predicted data
    sns.histplot(predictions_test_flat_masked, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.4,
                 label='Predicted CH')
    # ground truth data
    sns.histplot(targets_test_flat_masked, bins=30, kde=True, color='darkorange', edgecolor='black', alpha=0.4,
                 label='Ground Truth- CH')

    plt.xlabel('Canopy height')
    plt.ylabel('Frequency')
    plt.title('Predicted vs Ground Truth Canopy Height')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(targets_test_flat_masked, predictions_test_flat_masked, gridsize=50, cmap='inferno', mincnt=1, bins='log', alpha=0.8)
    cb = plt.colorbar(hb)
    cb.set_label('log10(count)')
    sns.regplot(x=targets_test_flat_masked, y=predictions_test_flat_masked, scatter=False, color='blue',
                line_kws={"linewidth": 1.5, "alpha": 0.7})
    plt.xlabel('Target Canopy Height ')
    plt.ylabel('Predicted Canopy Height ')
    plt.title('Target vs Predicted Canopy Heights (Test Data)')
    plt.tight_layout()
    plt.show()

    # More metrics:
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(targets_test_flat_masked, predictions_test_flat_masked, gridsize=50, cmap='inferno', mincnt=1,
                    bins='log', alpha=0.8)
    cb = plt.colorbar(hb)
    cb.set_label('log10(count)')
    sns.regplot(x=targets_test_flat_masked, y=predictions_test_flat_masked, scatter=False, color='blue',
                line_kws={"linewidth": 1.5, "alpha": 0.7})

    plt.text(0.05, 0.95, f"MAE: {test_mae:.4f}\nRMSE: {test_rmse:.4f}\nSamples: {len(predictions_test_flat_masked)}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Estimated height [m]')
    plt.ylabel('Target height (GEDI) [m]')
    plt.title('Target vs Predicted Canopy Heights (Test Data)')
    plt.tight_layout()
    plt.show()
