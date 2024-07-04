# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data
from typing import Dict, List, Optional, Sequence, Tuple
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch.optim as optim
import torch.nn.functional as F


from scale_hyperprior import ScaleHyperprior


class ScaleHyperpriorLightning(pl.LightningModule):
    def __init__(
        self,
        model: ScaleHyperprior,
        distortion_lambda,
    ):
        super().__init__()

        self.model = model
        self.distortion_lambda = distortion_lambda


    def forward(self, images):
        return self.model(images)
        
    def training_step(self, batch, batch_idx):
        
        images = batch

        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "train_loss": combined_loss.item(),
                "train_distortion_loss": distortion_loss.item(),
                "train_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        return {
            "loss": combined_loss,
           }


    def training_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('train_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

        # normal_imshow(self.model.rec_image[0].to('cpu').detach().numpy())
        # plt.show()

    def validation_step(self, batch, batch_idx):
        
        self.model.p_hyper_latent = .2
        self.model.p_latent = .2
        
        images = batch
        
        x_hat, y_likelihoods, z_likelihoods = self.model(images)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, images
        )
        self.log_dict(
            {
                "val_loss": combined_loss.item(),
                "val_distortion_loss": distortion_loss.item(),
                "val_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True, prog_bar=True, on_epoch=True, logger=True)

        self.model.p_hyper_latent = None
        self.model.p_latent = None

        return {
            "loss": combined_loss,
           }


    def validation_epoch_end(self, outs):
        loss_rec = torch.stack([x["loss"] for x in outs]).mean()
        self.log('val_combined_loss_epoch', loss_rec, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

        return {
                "optimizer": optimizer,
            }

        
    def rate_distortion_loss(self, reconstruction, latent_likelihoods,
                             hyper_latent_likelihoods, original,):
        
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width

        bits = (
            latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()
        ) / -math.log(2)
        
        bpp_loss = bits / num_pixels

        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss

        return bpp_loss, distortion_loss, combined_loss

