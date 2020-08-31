from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torchvision.models as models
from torch import nn

import os
import torch
import pandas as pd
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import tensorflow as tf
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import pdb

IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 64
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
BATCH_SIZE = 16
BUFFER_SIZE = 10000


BATCH_SIZE_SMALL = 2
BUFFER_SIZE_SMALL = 10

embedding_dim = 256
units = 512
# num_steps = len(text_data) // BATCH_SIZE

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1