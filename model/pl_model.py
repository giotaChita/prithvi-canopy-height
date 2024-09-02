import lightning as L
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from utils.config import pretrained_model_path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from prithvi.Prithvi import MaskedAutoencoderViT


class CHLighteningModule(L.LightningModule):
    def __init__(self, tile_size, patch_size, hparams, train_dataset, val_dataset, test_dataset):
        super(CHLighteningModule, self).__init__()

        self.save_hyperparameters(hparams)

        self.model = MaskedAutoencoderViT(img_size=tile_size, patch_size=patch_size,
                                 num_frames=1, tubelet_size=1, in_chans=6, embed_dim=768, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

        # Load pretrained weights
        self.load_pretrained_weights()
        # Freeze param except from decoder ch
        self.freezing_layers()

        # Initialize best metrics
        self.best_val_mae = float('inf')
        self.best_model_weights = None

        self.val_losses = []
        self.val_maes = []
        self.train_losses = []
        self.train_maes = []

        # Load dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.forward_loss(y_hat, y)

        y_hat_flat = y_hat.view(-1).detach().cpu().numpy()
        y_flat = y.view(-1).detach().cpu().numpy()
        # Mask
        mask = ~np.isnan(y_flat)

        if np.sum(mask) == 0:
            # Handle the case where all values are NaN
            mae = float('nan')
            rmse = float('nan')
            # r2 = float('nan')
        else:
            mae = mean_absolute_error(y_flat[mask], y_hat_flat[mask])
            rmse = np.sqrt(mean_squared_error(y_flat[mask], y_hat_flat[mask]))
            # r2 = r2_score(y_flat[mask], y_hat_flat[mask])

        # if loss/mae is nan take the previous value
        if torch.isnan(loss):
            loss = self.train_losses[-1] if self.train_losses else torch.tensor(0.0, requires_grad=True)
            mae = self.train_maes[-1] if self.train_maes else torch.tensor(0.0, requires_grad=True)

        return loss, mae, rmse #r2

    def training_step(self, batch, batch_idx):

        loss, mae, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', mae, prog_bar=True)
        self.train_losses.append(loss)
        self.train_maes.append(mae)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae,_ = self.common_step(batch, batch_idx)

        # Log loss and MAE
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_mae', mae, prog_bar=False)
        self.val_losses.append(loss.item())
        self.val_maes.append(mae)

        return loss

    def test_step(self, batch, batch_idx):
        loss, mae, rmse = self.common_step(batch, batch_idx)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)
        self.log('test_rmse', rmse, prog_bar=True)
        # self.log('test_r2', r2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(params, lr=self.hparams['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self.val_losses) if self.val_losses else float('inf')
        avg_val_mae = np.mean(self.val_maes) if self.val_maes else float('inf')

        self.log('average val_loss', avg_val_loss, prog_bar=True)
        self.log('average val_mae', avg_val_mae, prog_bar=True)

        # Check for best model
        if avg_val_mae < self.best_val_mae:
            self.best_val_mae = avg_val_mae
            self.best_model_weights = self.state_dict()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)

    def backward(self, loss, optimizer=None, optimizer_idx=None, *args, **kwargs):
        """Override the backward method to retain the graph during the backward pass."""
        if optimizer is not None:
            super().backward(loss, optimizer, optimizer_idx, retain_graph=True, *args, **kwargs)
        else:
            loss.backward(retain_graph=True)
    # def configure_optimizers(self):
    #     return optim.AdamW(self.parameters(),  self.hparams.lr)

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08),
    #         'monitor': 'average val_loss'
    #     }
    #
    #     # # scheduler1 = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
    #     # scheduler2 = get_linear_schedule_with_warmup(
    #     #     optimizer,
    #     #     num_warmup_steps=0,
    #     #     num_training_steps=self.hparams.num_epochs
    #     # )
    #     return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams['lr'])
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    #     return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'average val_loss'}}