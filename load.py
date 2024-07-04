# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torchvision import transforms
import seaborn as sns
import pandas as pd
import os
import torch.nn as nn
import torch.utils.data
import PIL
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM
from tqdm import tqdm
import glob
from ProgDTD import ScaleHyperpriorLightning, ScaleHyperprior
import yaml

device = 'cuda:0'

with open('params.yaml', "r") as yaml_file:
    config = yaml.safe_load(yaml_file)


def main():
    model = torch.load('weights.zip', map_location=torch.device('cpu')) 



if __name__ == "__main__":
    main()