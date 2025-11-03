# import torch
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visualize import visualize_sample_batch
from data import PascalVOCSegmentation




if __name__ == "__main__":
    data = PascalVOCSegmentation()
    loader = data.get_train_loader()
    visualize_sample_batch(loader)
    visualize_sample_batch(loader)
    visualize_sample_batch(loader)
