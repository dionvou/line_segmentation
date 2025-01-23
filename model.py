import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class LineSegmentationDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, valid=False, tile_size=(256, 256), stride=256):
        """
        Args:
            images_path (str): Path to the folder containing image files.
            labels_path (str): Path to the folder containing label files (binary masks).
            transform (callable, optional): Optional transform to be applied on a sample.
            tile_size (tuple): Size of the tiles to extract (height, width).
            stride (int): Stride for moving the window to extract overlapping tiles.
        """
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride

        if valid:
            self.image_files = glob.glob(os.path.join(images_path, '*.jpg'))[-2:]
            self.label_files = [os.path.join(labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.image_files]
        else:
            # Get list of image files (assuming they have the .jpg extension)
            self.image_files = glob.glob(os.path.join(images_path, '*.jpg'))[:-2]
            # Get list of corresponding label files (assuming they have the .png extension)
            self.label_files = [os.path.join(labels_path, os.path.splitext(os.path.basename(image_file))[0] + '.png')
                                for image_file in self.image_files]

    def __len__(self):
        # Total number of tiles in the dataset
        total_tiles = 0
        for image_file in self.image_files:
            img = Image.open(image_file)
            img_width, img_height = img.size

            # Calculate number of tiles in height and width
            tiles_in_height = (img_height - self.tile_size[0]) // self.stride + 1
            tiles_in_width = (img_width - self.tile_size[1]) // self.stride + 1

            total_tiles += tiles_in_height * tiles_in_width

        return total_tiles

    def __getitem__(self, idx):
        # Get the image and label index based on the total tiles
        img_idx = 0
        remaining_idx = idx
        while remaining_idx >= 0:
            img = Image.open(self.image_files[img_idx])
            label = Image.open(self.label_files[img_idx]).convert('L')

            img_width, img_height = img.size
            tiles_in_height = (img_height - self.tile_size[0]) // self.stride + 1
            tiles_in_width = (img_width - self.tile_size[1]) // self.stride + 1

            # Check if the tile falls within the current image
            if remaining_idx < tiles_in_height * tiles_in_width:
                break
            remaining_idx -= tiles_in_height * tiles_in_width
            img_idx += 1

        # Extract the row and column of the tile
        row_idx = remaining_idx // tiles_in_width
        col_idx = remaining_idx % tiles_in_width

        # Calculate the coordinates of the top-left corner of the tile
        top_left_y = row_idx * self.stride
        top_left_x = col_idx * self.stride

        # Crop the image and label to the tile size
        image_tile = img.crop((top_left_x, top_left_y, top_left_x + self.tile_size[1], top_left_y + self.tile_size[0]))
        label_tile = label.crop((top_left_x, top_left_y, top_left_x + self.tile_size[1], top_left_y + self.tile_size[0]))

        # Apply transformations if any (using albumentations or similar)
        if self.transform:
            augmented = self.transform(image=np.array(image_tile), mask=np.array(label_tile))
            image_tile = augmented['image']
            label_tile = augmented['mask']

        image_tile = image_tile/255
        label_tile = (label_tile > 0.5).float()  # Binary mask conversion

        return image_tile, label_tile


def get_transforms(data, cfg):
    if data == 'train':
        # Use values from cfg to define training augmentations
        transform = A.Compose([
            A.Rotate(limit=(-180, 180), p=cfg['train_augmentations']['rotate_prob']),
            # A.HorizontalFlip(p=cfg['train_augmentations']['flip_prob']),
            # A.VerticalFlip(p=cfg['train_augmentations']['flip_prob']),
            # A.RandomRotate90(p=cfg['train_augmentations']['rotate_prob']),
            A.RandomBrightnessContrast(p=cfg['train_augmentations']['brightness_contrast_prob']),
            # A.Resize(cfg['train_augmentations']['resize_height'], cfg['train_augmentations']['resize_width']),
            ToTensorV2()
        ], is_check_shapes=False)  # Disable shape consistency check
    elif data == 'valid':
        # Use values from cfg to define validation augmentations
        transform = A.Compose([
            ToTensorV2()
        ], is_check_shapes=False)  # Disable shape consistency check

    return transform


# best_model_state = None
def train(model, train_loader, valid_loader, epochs=10, lr=1e-4, device='cuda', patience=10):
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)  # Assuming binary segmentation
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Initialize variables for early stopping
    best_val_loss = float('inf')  # Start with a very high value
    epochs_without_improvement = 0  # Counter for early stopping
    best_model_state = None  # To save the best model state

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images).squeeze(1)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            # If validation loss improves, save the model and reset the patience counter
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # best_model_state = model.state_dict()  # Save the best model
            model_save_path = "best_model.pth"
            torch.save(best_model_state, model_save_path)

        else:
            # If validation loss doesn't improve, increment the counter
            epochs_without_improvement += 1
            print(f"Validation loss did not improve for {epochs_without_improvement} epochs.")

        # If we've hit the patience threshold, stop training early
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Validation loss has not improved for {patience} epochs.")
            break

    # Load the best model (with the lowest validation loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model based on validation loss.")

    print("Training complete.")

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionGate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention, self).__init__()

        # Contracting Path (Encoder)
        self.enc1 = self.conv_block_dil(in_channels, 64, dilation=1)
        self.enc2 = self.conv_block_dil(64, 128, dilation=1)
        self.enc3 = self.conv_block_dil(128, 256, dilation=1)
        self.enc4 = self.conv_block_dil(256, 512, dilation=1)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, dilation=1)

        # Expanding Path (Decoder)
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec_conv4 = self.conv_block(1024, 512, dilation=1)
        self.att4 = AttentionGate(512, 512, 256)

        self.upconv3 = self.upconv_block(512, 256)
        self.dec_conv3 = self.conv_block(512, 256, dilation=1)
        self.att3 = AttentionGate(256, 256, 128)

        self.upconv2 = self.upconv_block(256, 128)
        self.dec_conv2 = self.conv_block(256, 128, dilation=1)
        self.att2 = AttentionGate(128, 128, 64)

        self.upconv1 = self.upconv_block(128, 64)
        self.dec_conv1 = self.conv_block(128, 64, dilation=1)
        self.att1 = AttentionGate(64, 64, 1)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),  # Add BatchNorm after ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),   # Add BatchNorm after ReLU
            nn.ReLU(inplace=True)
        )

    def conv_block_dil(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),  # Add BatchNorm after ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),   # Add BatchNorm after ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),   # Add BatchNorm after ReLU
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Contracting Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding Path with Attention Gates
        up4 = self.upconv4(bottleneck)
        # att4 = self.att4(up4, enc4)  # Apply attention gate
        up4 = torch.cat([up4, enc4], dim=1)
        # up4 = torch.cat([up4, att4], dim=1)
        up4 = self.dec_conv4(up4)

        up3 = self.upconv3(up4)
        # att3 = self.att3(up3, enc3)  # Apply attention gate
        up3 = torch.cat([up3, enc3], dim=1)
        # up3 = torch.cat([up3, att3], dim=1)
        up3 = self.dec_conv3(up3)

        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2], dim=1)
        # att2 = self.att2(up2, enc2)  # Apply attention gate
        # up2 = torch.cat([up2, att2], dim=1)
        up2 = self.dec_conv2(up2)

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], dim=1)
        # att1 = self.att1(up1, enc1)  # Apply attention gate
        # up1 = torch.cat([up1, att1], dim=1)
        up1 = self.dec_conv1(up1)

        # Final Convolution
        out = self.final_conv(up1)
        return out

# RUNNING
# Path to your image and label directories
images_path = 'data/pretrain/img'
labels_path = 'data/pretrain/labels'

cfg = {
    'train_augmentations': {
        'flip_prob': 0.5,
        'rotate_prob': 0.6,
        'brightness_contrast_prob': 0.2,
        # 'resize_height': 512,
        # 'resize_width': 512
    },
    'valid_augmentations': {
        'to_tensor': True
    }
}
# Instantiate the dataset
dataset = LineSegmentationDataset(images_path, labels_path, transform=get_transforms('train', cfg))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print('len train: ', len(dataloader))


# Instantiate the dataset
valid_dataset = LineSegmentationDataset(images_path, labels_path, transform=get_transforms('valid', None), valid=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
print('len valid: ', len(valid_dataloader))


model = UNetWithAttention(in_channels=3, out_channels=1)
# Training loop
train(model, dataloader, valid_dataloader, epochs=2, lr=1e-4, device='cuda', patience=5)