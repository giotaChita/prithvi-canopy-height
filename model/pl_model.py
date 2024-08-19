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

        y_hat = y_hat.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        mask = ~np.isnan(y)

        mae = mean_absolute_error(y[mask], y_hat[mask])
        rmse = np.sqrt(mean_squared_error(y[mask], y_hat[mask]))
        r2 = r2_score(y[mask], y_hat[mask])

        # if loss/mae is nan take the previous value
        if torch.isnan(loss):
            loss = self.train_losses[-1]
            mae = self.train_maes[-1]

        return loss, mae, rmse, r2

    def training_step(self, batch, batch_idx):

        loss, mae,_,_ = self.common_step(batch, batch_idx)

        # Log loss and MAE
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', mae, prog_bar=True)
        # self.log_dict({'train_loss': loss, "train_mae": mae})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae,_,_ = self.common_step(batch, batch_idx)

        # Log loss and MAE
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        # self.log_dict({'val_loss': loss, "val_mae": mae})

        self.val_losses.append(loss.item())
        self.val_maes.append(mae)

        return loss

    def test_step(self, batch, batch_idx):
        loss, mae, rmse, r2 = self.common_step(batch, batch_idx)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)
        self.log('test_rmse', rmse, prog_bar=True)
        self.log('test_r2', r2, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        return pred


    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(),  self.hparams.lr)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self.val_losses) if self.val_losses else float('inf')
        avg_val_mae = np.mean(self.val_maes) if self.val_maes else float('inf')

        self.log('average val_loss', avg_val_loss, prog_bar=True)
        self.log('average val_mae', avg_val_mae, prog_bar=True)

        # Check for best model
        if avg_val_mae < self.best_val_mae:
            self.best_val_mae = avg_val_mae
            self.best_model_weights = self.state_dict()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.num_epochs
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)
