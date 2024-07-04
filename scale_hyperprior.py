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


from blocks import ImageAnalysis, HyperAnalysis, HyperSynthesis, ImageSynthesis


class ScaleHyperprior(nn.Module):
    def __init__(
        self,
        network_channels: Optional[int] = None,
        compression_channels: Optional[int] = None,
        image_analysis: Optional[nn.Module] = None,
        image_synthesis: Optional[nn.Module] = None,
        image_bottleneck: Optional[nn.Module] = None,
        hyper_analysis: Optional[nn.Module] = None,
        hyper_synthesis: Optional[nn.Module] = None,
        hyper_bottleneck: Optional[nn.Module] = None,
        progressiveness_range: Optional[List] = None,
    ):
        super().__init__()
        self.image_analysis = ImageAnalysis(network_channels, compression_channels)  
        self.hyper_analysis = HyperAnalysis(network_channels, compression_channels) 
        self.hyper_synthesis = HyperSynthesis(network_channels, compression_channels)  
        self.image_synthesis = ImageSynthesis(network_channels, compression_channels)
        
        self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)
        self.image_bottleneck = GaussianConditional(scale_table=None)
        self.progressiveness_range = progressiveness_range
        self.p_hyper_latent = None
        self.p_latent = None
        
    def forward(self, images):
            
        self.latent = self.image_analysis(images)
        self.hyper_latent = self.hyper_analysis(self.latent)
        
        #---***---#
        self.latent = self.rate_less_latent(self.latent)
        self.hyper_latent = self.rate_less_hyper_latent(self.hyper_latent)
        #---***---#

        
        self.noisy_hyper_latent, self.hyper_latent_likelihoods = self.hyper_bottleneck(
            self.hyper_latent
        )

        self.scales = self.hyper_synthesis(self.noisy_hyper_latent)
        self.noisy_latent, self.latent_likelihoods = self.image_bottleneck(self.latent, self.scales)
        
        #---***---#
        self.latent_likelihoods = self.drop_zeros_likelihood(self.latent_likelihoods, self.latent)
        self.hyper_latent_likelihoods = self.drop_zeros_likelihood(self.hyper_latent_likelihoods, self.hyper_latent)
        #---***---#
        
        self.reconstruction = self.image_synthesis(self.noisy_latent)

        self.rec_image = self.reconstruction.detach().clone()

        return self.reconstruction, self.latent_likelihoods, self.hyper_latent_likelihoods



    def rate_less_latent(self, data):
        self.save_p = []
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_latent:
                # p shows the percentage of keeping
                p = self.p_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1],1)[0]
                self.save_p.append(p)

            if p == 1.0:
                pass            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data
    
    def rate_less_hyper_latent(self, data):
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_hyper_latent:
                # p shows the percentage of keeping
                p = self.p_hyper_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
                p = self.save_p[i]
            if p == 1.0:
                pass
            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data

    def drop_zeros_likelihood(self, likelihood, replace):
        temp_data = likelihood.clone()
        temp_data = torch.where(
            replace == 0.0,
            torch.cuda.FloatTensor([1.0])[0],
            likelihood,
        )
        return temp_data
