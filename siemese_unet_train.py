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

class SiameseExhaustiveDataset(Dataset):
    def __init__(self, support_images_path, support_labels_path, images_path, labels_path, transform=None,transform_valid=None,valid=False, tile_size=(512, 512), stride=512):
        """
        Creates dataset where each validation tile is paired with all training tiles.
        """
        self.support_image_files = glob.glob(os.path.join(support_images_path, '*.jpg'))
        self.support_label_files = [os.path.join(support_labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.support_image_files]
        # self.support_label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(support_labels_path, '*.png'))}

        self.image_files = glob.glob(os.path.join(images_path, '*.jpg'))[:2]
        self.label_files = [os.path.join(labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.image_files]
        # self.label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(labels_path, '*.png'))}

        self.transform = transform
        self.transform_valid = transform_valid
        self.tile_size = tile_size
        self.stride = stride
        if valid:
            # # HARDCODED
            # self.image_files = self.image_files[1:3]
            # # Precompute all  tiles
            # self.tiles = []
            # for img_idx, image_file in enumerate(self.image_files):
            #     query_img = Image.open(image_file)
            #     query_label = Image.open(self.label_files[os.path.splitext(os.path.basename(image_file))[0]]).convert('L')
            #     tiles, label_tiles = self.get_tiles(query_img, query_label)
            #     for tile, label_tile in zip(tiles, label_tiles):
            #         self.tiles.append((img_idx, tile, label_tile))

            # # Precompute all support tiles
            # self.support_tiles = []
            # for img_idx, image_file in enumerate(self.support_image_files):
            #     support_img = Image.open(image_file)
            #     support_label = Image.open(self.support_label_files[os.path.splitext(os.path.basename(image_file))[0]]).convert('L')
            #     tiles, label_tiles = self.get_tiles(support_img, support_label)
            #     for tile, label_tile in zip(tiles, label_tiles):
            #         self.support_tiles.append((img_idx, tile, label_tile))
                        # Precompute all tiles for validation images
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
            # # HARDCODED
            # self.image_files = self.image_files[1:3]
            # # Precompute all query tiles
            # self.tiles = []
            # for img_idx, image_file in enumerate(self.image_files):
            #     query_img = Image.open(image_file)
            #     query_label = Image.open(self.label_files[os.path.splitext(os.path.basename(image_file))[0]]).convert('L')
            #     tiles, label_tiles = self.get_tiles(query_img, query_label)
            #     for tile, label_tile in zip(tiles, label_tiles):
            #         self.tiles.append((img_idx, tile, label_tile))

            # # Precompute all support tiles
            # self.support_tiles = []
            # for img_idx, image_file in enumerate(self.support_image_files):
            #     support_img = Image.open(image_file)
            #     support_label = Image.open(self.support_label_files[os.path.splitext(os.path.basename(image_file))[0]]).convert('L')
            #     tiles, label_tiles = self.get_tiles(support_img, support_label)
            #     for tile, label_tile in zip(tiles, label_tiles):
            #         self.support_tiles.append((img_idx, tile, label_tile))
                        # Precompute all tiles for validation images
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
        for _, tile, label_tile in self.support_tiles[3:9]:
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
            A.InvertImg(p=0.2),  # Inverts pixel values like a negative filter
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
            # A.Resize(cfg['valid_augmentations']['resize_height'], cfg['valid_augmentations']['resize_width']),
            ToTensorV2(),
        ])
    return transform


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class CrossAttentionWithMask(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionWithMask, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.threshold = nn.Parameter(torch.tensor(0.8))  # Learnable threshold
        self.temperature = nn.Parameter(torch.tensor(10.0))  # Learnable scaling factor

    def forward(self, query, support):
        B, C, H, W = query.shape  # B = 1 (single query image)
        N = support.shape[0]  # Number of support images

        # Reshape query to (B, H*W, C)
        query = query.view(B, C, H * W).permute(0, 2, 1)  # (1, H*W, C)

        # Reshape support to (N, C, H*W) and expand for batch dim
        support = support.view(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)

        # Expand query to match support batch size
        query = query.repeat(N, 1, 1)  # (N, H*W, C)

        # Project query and support
        Q = self.query_proj(query)   # (N, H*W, C)
        K = self.key_proj(support)   # (N, H*W, C)
        V = self.value_proj(support) # (N, H*W, C)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (N, H*W, H*W)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Ensure threshold is in a valid range
        threshold = torch.sigmoid(self.threshold)  # Ensures value is between 0 and 1
        temperature = F.softplus(self.temperature)  # Ensures it's positive

        # Compute similarity mask with a differentiable soft threshold
        max_attn_values = attn_probs.max(dim=-1).values  # (N, H*W)
        similarity_mask = torch.sigmoid((max_attn_values - threshold) * temperature).unsqueeze(-1)  # (N, H*W, 1)

        # Apply attention
        attended_features = torch.matmul(attn_probs, V)  # (N, H*W, C)
        masked_features = attended_features * similarity_mask  # (N, H*W, C)

        return masked_features.permute(0, 2, 1).view(N, C, H, W)

class SiameseUNetResNet50(nn.Module):
    def __init__(self, out_channels):
        super(SiameseUNetResNet50, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        self.cross_attention = CrossAttentionWithMask(2048)

        self.upconv4 = self.upconv_block(2048, 1024)
        self.dec4 = self.conv_block(2048, 1024)
        self.upconv3 = self.upconv_block(1024, 512)
        self.dec3 = self.conv_block(1024, 512)
        self.upconv2 = self.upconv_block(512, 256)
        self.dec2 = self.conv_block(512, 256)
        self.upconv1 = self.upconv_block(256, 64)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, support_img, query_img):
        enc1_s, enc1_q = self.enc1(support_img), self.enc1(query_img)
        enc2_s, enc2_q = self.enc2(enc1_s), self.enc2(enc1_q)
        enc3_s, enc3_q = self.enc3(enc2_s), self.enc3(enc2_q)
        enc4_s, enc4_q = self.enc4(enc3_s), self.enc4(enc3_q)
        enc5_s, enc5_q = self.enc5(enc4_s), self.enc5(enc4_q)

        attended_features = self.cross_attention(enc5_q, enc5_s)
        attended_features = attended_features.max(dim=0, keepdim=True)[0] # (N,C,H,W)

        fusion = torch.cat([enc5_q * attended_features], dim=1)

        up4 = self.upconv4(fusion)
        up4 = torch.cat([up4, enc4_q], dim=1)
        up4 = self.dec4(up4)
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, enc3_q], dim=1)
        up3 = self.dec3(up3)
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2_q], dim=1)
        up2 = self.dec2(up2)
        up1 = self.upconv1(up2)
        enc1_q_resized = F.interpolate(enc1_q, size=up1.shape[2:], mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, enc1_q_resized], dim=1)
        up1 = self.dec1(up1)

        output = F.interpolate(up1, size=(query_img.shape[2], query_img.shape[3]), mode="bilinear", align_corners=False)
        return self.final_conv(output)

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
train_dataset = SiameseExhaustiveDataset(images_path, labels_path, images_path, labels_path, transform=get_transforms('train', cfg),transform_valid=get_transforms('train', cfg))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

val_dataset = SiameseExhaustiveDataset(images_path, labels_path, val_images_path, val_labels_path,valid=True, transform=get_transforms('train', cfg),transform_valid=get_transforms('valid', cfg))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = SiameseUNetResNet50(out_channels=1)

criterion = nn.BCEWithLogitsLoss()  # Pixel-wise classification loss
print(len(train_dataloader))
print(len(val_dataloader))


utils.train_model(model, train_dataloader, val_dataloader, criterion, epochs=1, lr=5e-4, patience=10, path='model_weights/best_model.pth')
preds = utils.get_predictions(model,val_dataloader)
reconstructed = utils.reconstruct(val_dataloader,preds,'images')
print(len(reconstructed))
thresholded = utils.thresholding_images(reconstructed, 0.3)
denoised_images = [utils.remove_small_objects(img, min_size=100) for img in thresholded]
print(utils.evaluate_IU(val_dataloader, denoised_images, 0.5))

