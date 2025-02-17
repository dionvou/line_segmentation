import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import models
import glob
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import f1_score, fbeta_score, average_precision_score
import random
import utils
import models


class SiameseExhaustiveDataset(Dataset):
    def __init__(self, support_images_path, support_labels_path, images_path, labels_path, transform=None,transform_valid=None,valid=False, tile_size=(512, 512), stride=512):
        """
        Creates dataset where each validation tile is paired with all training tiles.
        """
        self.support_image_files = glob.glob(os.path.join(support_images_path, '*.jpg'))[:3]
        self.support_label_files = [os.path.join(support_labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.support_image_files]
        # self.support_label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(support_labels_path, '*.png'))}

        self.image_files = glob.glob(os.path.join(images_path, '*.jpg'))[:3]
        self.label_files = [os.path.join(labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.image_files]
        # self.label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(labels_path, '*.png'))}

        self.transform = transform
        self.transform_valid = transform_valid
        self.tile_size = tile_size
        self.stride = stride
        if valid:
            self.tiles = []
            for img_idx, (image_file, label_file) in enumerate(zip(self.image_files, self.label_files)):
                query_img = Image.open(image_file)
                query_label = Image.open(label_file).convert('L')
                tiles, label_tiles = self.get_tiles(query_img, query_label)
                for tile, label_tile in zip(tiles, label_tiles):
                    self.tiles.append((img_idx, tile, label_tile))

            # Precompute all tiles for support images
            self.support_tiles = []
            for img_idx, (image_file, label_file) in enumerate(zip(self.support_image_files, self.support_label_files)):
                support_img = Image.open(image_file)
                support_label = Image.open(label_file).convert('L')
                tiles, label_tiles = self.get_tiles(support_img, support_label)
                for tile, label_tile in zip(tiles, label_tiles):
                    self.support_tiles.append((img_idx, tile, label_tile))

        else:
            self.tiles = []
            for img_idx, (image_file, label_file) in enumerate(zip(self.image_files, self.label_files)):
                query_img = Image.open(image_file)
                query_label = Image.open(label_file).convert('L')
                tiles, label_tiles = self.get_tiles(query_img, query_label)
                for tile, label_tile in zip(tiles, label_tiles):
                    self.tiles.append((img_idx, tile, label_tile))

            # Precompute all tiles for support images
            self.support_tiles = []
            for img_idx, (image_file, label_file) in enumerate(zip(self.support_image_files, self.support_label_files)):
                support_img = Image.open(image_file)
                support_label = Image.open(label_file).convert('L')
                tiles, label_tiles = self.get_tiles(support_img, support_label)
                for tile, label_tile in zip(tiles, label_tiles):
                    self.support_tiles.append((img_idx, tile, label_tile))


    def __len__(self):
        return len(self.tiles)

    def get_tiles(self, img, label):
        img_width, img_height = img.size
        pad_w = (self.stride - (img_width % self.stride)) % self.stride
        pad_h = (self.stride - (img_height % self.stride)) % self.stride

        img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)
        label = ImageOps.expand(label, (0, 0, pad_w, pad_h), fill=0)
        padded_width, padded_height = img.size

        tiles = []
        label_tiles = []
        for y in range(0, padded_height - self.tile_size[0] + 1, self.stride):
            for x in range(0, padded_width - self.tile_size[1] + 1, self.stride):
                img_tile = img.crop((x, y, x + self.tile_size[1], y + self.tile_size[0]))
                label_tile = label.crop((x, y, x + self.tile_size[1], y + self.tile_size[0]))
                tiles.append(img_tile)
                label_tiles.append(label_tile)

        return tiles, label_tiles

    def __getitem__(self, idx):
        img_idx, query_tile, query_label_tile = self.tiles[idx]

        support_tiles = []
        support_label_tiles = []
        for _, tile, label_tile in self.support_tiles[:]:
            support_tiles.append(tile)
            support_label_tiles.append(label_tile)

        if self.transform:
            transformed = [self.transform(image=np.array(tile), mask=np.array(label))
                          for tile, label in zip(support_tiles, support_label_tiles)]

            support_tiles = [aug['image'] for aug in transformed]
            support_label_tiles = [aug['mask'] for aug in transformed]

        if self.transform_valid:

            query_transformed = self.transform_valid(image=np.array(query_tile), mask=np.array(query_label_tile))
            query_tile = query_transformed['image']
            query_label_tile = query_transformed['mask']

        support_tiles = [tile / 255.0 for tile in support_tiles]
        query_tile = query_tile / 255.0
        support_label_tiles = [(label > 0.5).float() for label in support_label_tiles]
        query_label_tile = (query_label_tile > 0.5).float()

        return support_tiles, support_label_tiles, query_tile, query_label_tile

def get_transforms(data, cfg):
    if data == "train":
        transform = A.Compose([
            # A.Rotate(limit=(-90, 90), p=cfg['train_augmentations']['rotate_prob']),
            # A.HorizontalFlip(p=cfg['train_augmentations']['flip_prob']),
            # A.VerticalFlip(p=cfg['train_augmentations']['flip_prob']),
            # A.RandomRotate90(p=cfg['train_augmentations']['rotate_prob']),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Randomly changes brightness, contrast, saturation, hue
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Adjusts hue, saturation, and value
            # A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Randomly changes gamma to simulate different lighting conditions
            # A.Equalize(p=0.3),  # Equalizes histogram of image (helps with contrast)
            # A.Solarize(threshold=128, p=0.2),  # Inverts colors above a certain intensity
            A.ToGray(p=0.3),  # Converts image to grayscale randomly
            # A.InvertImg(p=0.2),  # Inverts pixel values like a negative filter
            # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            # A.GaussNoise(var_limit=(10, 50), mean=0, p=0.5),  
            A.RandomBrightnessContrast(p=cfg['train_augmentations']['brightness_contrast_prob']),
            A.RandomResizedCrop(
                size = (cfg['train_augmentations']['resize_height'], cfg['train_augmentations']['resize_width']),
                scale=(0.5, 1),
                p=1  # Ensure this transformation is always applied during training
            ),
            
            ToTensorV2(),
        ])
    elif data in ["valid", "test"]:
        transform = A.Compose([
            ToTensorV2(),
        ])
    return transform


# Config
cfg = {
    "train_augmentations": {
        "flip_prob": 0.5,
        "rotate_prob": 0.5,
        "brightness_contrast_prob": 0.4,
        "resize_height": 512,
        "resize_width": 512
    },
    "valid_augmentations": {
        "resize_height": 512,
        "resize_width": 512
    }
}

images_path =  'data/Latin2/img-Latin2/training/'
labels_path = 'data/Latin2/text-line-gt-Latin2/training/'

val_images_path =  'data/Latin2/img-Latin2/validation/'
val_labels_path = 'data/Latin2/text-line-gt-Latin2/validation/'

# Update DataLoader
train_dataset = SiameseExhaustiveDataset(images_path, labels_path, images_path, labels_path, transform=get_transforms('valid', cfg),transform_valid=get_transforms('train', cfg))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

val_dataset = SiameseExhaustiveDataset(images_path, labels_path, val_images_path, val_labels_path,valid=True, transform=get_transforms('valid', cfg),transform_valid=get_transforms('valid', cfg))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = models.get_model('resnet50','cross_correlation')

# criterion = nn.BCEWithLogitsLoss()  # Pixel-wise classification loss
criterion = utils.focal_loss()
# utils.load_model_weights(model)

utils.train_model(model, train_dataloader, val_dataloader, criterion, epochs=60, lr=5e-4, patience=5, path='model_weights/best_model.pth')
preds = utils.get_predictions(model,val_dataloader)
reconstructed = utils.reconstruct(val_dataloader,preds,'images')
thresholded = utils.thresholding_images(reconstructed, 0.3)
denoised_images = [utils.remove_small_objects(img, min_size=300) for img in thresholded]
eroded_images = [utils.erode(img, kernel_size=3, iterations=1) for img in denoised_images]
dilated_images = [utils.dilate(img, kernel_size=3, iterations=1) for img in eroded_images]

print(utils.evaluate_IU(val_dataloader, denoised_images, 0.5))
print(utils.evaluate_IU(val_dataloader, dilated_images, 0.5))
utils.save_images(dilated_images)


