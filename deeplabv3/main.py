import deeplabv3
import torch
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt


model=deeplabv3.model_load()

#print(model)
print('model is ',model)

