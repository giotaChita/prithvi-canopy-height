from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
from utils.config import pretrained_model_path, best_model_path
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from copy import deepcopy
from utils.utils import compute_metrics

def train_val_loop(model,device, batch_size, patch_size, tile_size, train_loader, val_loader ):

        # Load pretrained weights
        pretrained_weights = torch.load(pretrained_model_path)

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_weights.items() if
                           k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(pretrained_dict)

        # Load the new state dict
        model.load_state_dict(model_dict, strict=False)

        # optimizer = torch.optim.Adam(model.canopy_height_head.parameters(), lr=1e-3, eps=1e-8)
        optimizer = optim.AdamW(model.canopy_height_head.parameters(), lr=1e-3, eps=1e-8)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.canopy_height_head.parameters():
            param.requires_grad = True

        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        num_param = {sum(p.numel() for p in model.parameters() if p.requires_grad)}

        num_epochs = 500
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler()

        # lists to store loss values
        train_losses = []
        val_losses = []
        mae_list_val = []
        mae_list_train = []
        # variables to track the best model
        best_val_loss = float('inf')

        # Comet
        experiment = Experiment(
            api_key="YZiwsYqIN87kijoaS5atmnqqz",
            project_name="prithvi_original-ch",
            workspace="tagio"

        )

        # Log hyperparameters to Comet ML
        experiment.log_parameters({
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'num_epochs': num_epochs,
            'tile_size': tile_size,
            'patch_size': patch_size,
            'embed_dim': 768
        })

        model.to(device)

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
                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    pred = model(inputs)
                    loss = model.forward_loss(pred, labels)

                    predictions_list_train.append(pred.detach().cpu().numpy())
                    targets_list_train.append(labels.detach().cpu().numpy())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                # before_lr = optimizer.param_groups[0]["lr"]
                scaler.update()
                # scheduler.step()
                running_loss += loss.item()
                count += 1

            predictions_train = np.concatenate(predictions_list_train, axis=0)
            targets_train = np.concatenate(targets_list_train, axis=0)

            # Reshape to (batch_size * H * W,)
            predictions_train = predictions_train.reshape(-1)
            targets_train = targets_train.reshape(-1)

            mask_train = ~np.isnan(targets_train)

            # Filter predictions and targets using the mask
            filtered_predictions_train = predictions_train[mask_train]
            filtered_targets_train = targets_train[mask_train]

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Mean Absolute Error (MAE)
            mae_train = mean_absolute_error(filtered_targets_train, filtered_predictions_train)
            mae_list_train.append(mae_train)

            # Validation loop
            model.eval()
            val_loss = 0.0
            predictions_list_val = []
            targets_list_val = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = model(images)
                    loss = model.forward_loss(pred, labels)
                    val_loss += loss.item()

                    predictions_list_val.append(pred.detach().cpu().numpy())
                    targets_list_val.append(labels.detach().cpu().numpy())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            predictions_val = np.concatenate(predictions_list_val, axis=0)
            targets_val = np.concatenate(targets_list_val, axis=0)

            # Reshape to (batch_size * H * W,)
            predictions_val = predictions_val.reshape(-1)
            targets_val = targets_val.reshape(-1)

            mask_val = ~np.isnan(targets_val)

            # Filter predictions and targets using the mask
            filtered_predictions_val = predictions_val[mask_val]
            filtered_targets_val = targets_val[mask_val]

            # Mean Absolute Error (MAE)
            mae_val = mean_absolute_error(filtered_targets_val, filtered_predictions_val)
            mae_list_val.append(mae_val)

            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mean_squared_error(filtered_targets_val, filtered_predictions_val))

            # R-squared
            r2 = r2_score(filtered_targets_val, filtered_predictions_val)

            # Print the metrics
            print(f"Mean Absolute Error (MAE): {mae_val:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"R-squared: {r2:.4f}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Log metrics to Comet ML
            experiment.log_metric('train_loss', train_loss, step=epoch + 1)
            experiment.log_metric('val_loss', val_loss, step=epoch + 1)
            experiment.log_metric('train_mae', mae_train, step=epoch + 1)
            experiment.log_metric('val_mae', mae_val, step=epoch + 1)

            # Find best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())

                # # Log the best model checkpoint to Comet ML
                # log_model(experiment, model=model, model_name="BestModel")

        # Save the best model
        torch.save(best_model_state, best_model_path)

        # Plot the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        # plt.savefig(os.path.join(save_dir, f'{num_epochs}_{embed_dims}_{batch_size}_lr1e3_AOI1_training_validation_loss.png'))
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), mae_list_train, label='Train MAE')
        plt.plot(range(1, num_epochs + 1), mae_list_val, label='VAL MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        # plt.savefig(os.path.join(save_dir, f'{num_epochs}_{embed_dims}_{batch_size}_lr1e3_AOI1_training_validation_mae.png'))
        plt.show()

        #comet
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

            # Collect predictions and targets for metrics calculation
            predictions_list.append(pred.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())

    test_loss /= len(test_loader)

    # Compute metrics for test set
    predictions_test = np.concatenate(predictions_list, axis=0).squeeze()
    targets_test = np.concatenate(targets_list, axis=0).squeeze()
    test_mae, test_rmse, test_r2 = compute_metrics(predictions_test, targets_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test R-squared: {test_r2:.4f}")

