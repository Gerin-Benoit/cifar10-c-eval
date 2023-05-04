import numpy as np
import os
import PIL
import torch
import torchvision
import random

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets

from utils import load_txt

corruptions = load_txt('./src/corruptions.txt')


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str, severity: int,
                 transform=None, target_transform=None):
        assert name in corruptions
        assert severity in range(1, 6)  # severity levels are from 1 to 5
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        # Load the entire dataset and select images for the desired severity level
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        num_images_per_severity = len(self.data) // 5
        start_index = (severity - 1) * num_images_per_severity
        end_index = start_index + num_images_per_severity
        self.data = self.data[start_index:end_index]
        self.targets = self.targets[start_index:end_index]

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)

'''
class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)
'''

def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)