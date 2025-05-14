import codecs
import os
import os.path
import shutil
import string
import sys
import pandas as pd
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
from torchvision import datasets
from torchvision.io import read_image
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image


class BuildingDataset(datasets.VisionDataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            label_file,
            image_file,
            img_dir,
            # root=None,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None) -> None:
        # super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.img_labels = pd.read_csv(label_file)
        self.img_names = pd.read_csv(image_file)
        self.img_dir = img_dir
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        img_path = "%s/%s" % (self.img_dir, self.img_names.iloc[idx, 0])
        # print(img_path)
        img = read_image(img_path)
        # print(img.shape)
        image = Image.fromarray(img.numpy(), mode="RGB")
        # print(type(image))
        # plt.imshow(image.T)
        # plt.show()
        label = self.img_labels.iloc[idx, 0]
        if self.transforms:
            image = self.transforms(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label