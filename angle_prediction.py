import os
import torch
from torch.autograd import Variable
from Auto_enco import MyAuto
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


torch.cuda.set_device(0)
torch.manual_seed(1)

# Super parameters
EPOCH = 100
BATCH_SIZE = 1
LR = 1e-3
is_shuffle = False