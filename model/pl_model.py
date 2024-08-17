from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import lightning as L

import torch.nn as nn
import torch.optim as optim
from utils.config import pretrained_model_path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from prithvi.Prithvi import MaskedAutoencoderViT


class CHLighteningModule(L.LightningModule):
    def __init__(self, tile_size, patch_size, hparams, train_len):
        super(CHLighteningModule, self).__init__()

        self.save_hyperparameters(hparams)

        self.model = MaskedAutoencoderViT(img_size=tile_size, patch_size=patch_size,
                                 num_frames=1, tubelet_size=1, in_chans=6, embed_dim=768, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

        # Load pretrained weights
        self.load_pretrained_weights()
        # Freeze param
        self.freezing_layers()

        # Lists to store loss and MAE values
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []

        # Datasets
        self.train_len = train_len

        # Initialize best metrics
        self.best_val_mae = float('inf')
        self.best_model_weights = None

        # # Comet
        # self.logger = logger
        # if logger:
        #     logger.experiment.log_parameters(hparams)

    def load_pretrained_weights(self):
        pretrained_weights = torch.load(pretrained_model_path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)

    def freezing_layers(self):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze decoder ch head parameters
        for param in self.model.canopy_height_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def calculate_mae(self, y_hat, y):
        y_hat = y_hat.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        mask = ~np.isnan(y)
        return mean_absolute_error(y[mask], y_hat[mask])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.forward_loss(y_hat, y)
        mae = self.calculate_mae(y_hat, y)

        if torch.isnan(loss):
            loss = self.train_losses[-1]
            mae = self.train_maes[-1]

        # Log loss and MAE
        self.log('train_loss', loss)  #, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae)  #, on_step=False, on_epoch=True, prog_bar=True)

        # self.log('train_loss', loss, prog_bar=True)
        # self.log('train_mae', mae, prog_bar=True)
        #
        self.train_losses.append(loss.item())
        self.train_maes.append(mae)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.forward_loss(y_hat, y)

        # Compute MAE
        mae = self.calculate_mae(y_hat, y)

        if torch.isnan(loss):
            loss = self.val_losses[-1]
            mae = self.val_maes[-1]

        # Log loss and MAE
        self.log('val_loss', loss) #, loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae) #, mae, on_step=False, on_epoch=True, prog_bar=True)

        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_mae', mae, prog_bar=True)

        self.val_losses.append(loss.item())
        self.val_maes.append(mae)

        return loss

    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1),
            'interval': 'epoch',  # or 'step' depending on your use case
            'frequency': 1,
            'reduce_lr_on_plateau': False  # Optional depending on scheduler
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self.val_losses) if self.val_losses else float('inf')
        avg_val_mae = np.mean(self.val_maes) if self.val_maes else float('inf')

        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_mae', avg_val_mae, prog_bar=True)

        # Check for best model
        if avg_val_mae < self.best_val_mae:
            self.best_val_mae = avg_val_mae
            self.best_model_weights = self.state_dict()
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log_model('best_model', self.best_model_weights)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.forward_loss(y_hat, y)

        # Compute metrics
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
        y = y.detach().cpu().numpy().reshape(-1)
        mask = ~np.isnan(y)
        mae = mean_absolute_error(y[mask], y_hat[mask])
        rmse = np.sqrt(mean_squared_error(y[mask], y_hat[mask]))
        r2 = r2_score(y[mask], y_hat[mask])

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)
        self.log('test_rmse', rmse, prog_bar=True)
        self.log('test_r2', r2, prog_bar=True)

        return loss
