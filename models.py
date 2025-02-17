import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights


class CrossAttentionWithMask(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionWithMask, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.threshold = nn.Parameter(torch.tensor(0.8))  # Learnable threshold
        self.temperature = nn.Parameter(torch.tensor(10.0))  # Learnable scaling factor

    def forward(self, query, support, support_labels):
        B, C, H, W = query.shape  # B = 1 (single query image)
        N = support.shape[0]  # Number of support images

        # Downsample support_labels to match bottleneck resolution
        support_labels = F.interpolate(support_labels.unsqueeze(1).float(), size=(H, W), mode='nearest')  # (N, 1, H, W)
        support_mask = (support_labels > 0.5).float()  # Convert to binary mask

        # Reshape tensors for attention computation
        query = query.view(B, C, H * W).permute(0, 2, 1)  # (1, H*W, C)
        support = support.view(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)
        support_mask = support_mask.view(N, 1, H * W).permute(0, 2, 1)  # (N, H*W, 1)

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

        # Apply support mask to attention
        # attn_probs = attn_probs * support_mask  # Zero out background areas

        # Apply attention
        attended_features = torch.matmul(attn_probs, V)  # (N, H*W, C)
        masked_features = attended_features * similarity_mask  # (N, H*W, C)

        return masked_features.permute(0, 2, 1).view(N, C, H, W)
    
class DepthWiseFeatureCorrelation(nn.Module):
    def __init__(self):
        super(DepthWiseFeatureCorrelation, self).__init__()

    def forward(self, query, support):
        """
        Compute depth-wise cross-correlation between query and support images.
        query: (1, C, H, W)  # Single query image
        support: (B, C, H, W)  # B support images
        Returns: (B, H, W) correlation map
        """
        B, C, H, W = support.shape

        # Expand query to match batch size of support images
        query = query.expand(B, -1, -1, -1)  # (B, C, H, W)

        # Ensure query acts as depth-wise convolution kernel
        query = query.view( C, B, H, W)  # (B, C, 1, H, W)
        support = support.view(1, C, H, W)  # (B, C, 1, H, W)

        # Perform depth-wise cross-correlation
        correlation = F.conv2d(support, query, groups=C, padding=0)  # (B, C, 1, 1)

        # Sum over channels to get (B, 1, 1)
        correlation = correlation.sum(dim=1)  # (B, 1, 1)

        # Expand to (B, H, W)
        correlation = correlation.expand(B, H, W)
    
class SiameseUNetResNet50(nn.Module):
    def __init__(self, out_channels, mechanism='cross_attention'):
        super(SiameseUNetResNet50, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        if mechanism =='cross_attention':
            self.mechanism = CrossAttentionWithMask(2048)
        elif mechanism == 'cross_correlation':
            self.mechanism = DepthWiseFeatureCorrelation(2048)

        self.upconv4 = self.upconv_block(2048, 1024)
        self.dec4 = self.conv_block(2048, 1024)
        self.upconv3 = self.upconv_block(1024, 512)
        self.dec3 = self.conv_block(1024, 512)
        self.upconv2 = self.upconv_block(512, 256)
        self.dec2 = self.conv_block(512, 256)
        self.upconv1 = self.upconv_block(256, 64)
        self.dec1 = self.conv_block(128, 64)
        self.upconv = self.upconv_block(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

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

    def forward(self, support_img, query_img, support_masks):
        enc1_s, enc1_q = self.enc1(support_img), self.enc1(query_img)
        enc2_s, enc2_q = self.enc2(enc1_s), self.enc2(enc1_q)
        enc3_s, enc3_q = self.enc3(enc2_s), self.enc3(enc2_q)
        enc4_s, enc4_q = self.enc4(enc3_s), self.enc4(enc3_q)
        enc5_s, enc5_q = self.enc5(enc4_s), self.enc5(enc4_q)

        attended_features = self.mechanism(enc5_q, enc5_s, support_masks)
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
        output = self.upconv(up1)
        return self.final_conv(output)
    
class SiameseUNetVGG16(nn.Module):
    def __init__(self, out_channels, mechanism='cross_attention'):
        super(SiameseUNetVGG16, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Encoder (VGG-16 feature extractor)
        self.enc1 = vgg[:4]   # Conv1_1 -> Conv1_2 -> ReLU -> MaxPool
        self.enc2 = vgg[4:9]  # Conv2_1 -> Conv2_2 -> ReLU -> MaxPool
        self.enc3 = vgg[9:16]  # Conv3_1 -> Conv3_2 -> Conv3_3 -> ReLU -> MaxPool
        self.enc4 = vgg[16:23] # Conv4_1 -> Conv4_2 -> Conv4_3 -> ReLU -> MaxPool
        self.enc5 = vgg[23:30] # Conv5_1 -> Conv5_2 -> Conv5_3 -> ReLU -> MaxPool
        
        if mechanism =='cross_attention':
            self.mechanism = CrossAttentionWithMask(512)
        elif mechanism == 'cross_correlation':
            self.mechanism = DepthWiseFeatureCorrelation(512)

        # Decoder (upsampling + skip connections)
        self.upconv4 = self.upconv_block(512, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final segmentation output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        """Upsampling block using transposed convolution."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        """Double convolution block with BatchNorm and Dropout."""
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

    def forward(self, support_img, query_img, support_masks):
        """Forward pass for the Siamese U-Net model using VGG-16."""

        # Encode support and query images
        enc1_s, enc1_q = self.enc1(support_img), self.enc1(query_img)
        enc2_s, enc2_q = self.enc2(enc1_s), self.enc2(enc1_q)
        enc3_s, enc3_q = self.enc3(enc2_s), self.enc3(enc2_q)
        enc4_s, enc4_q = self.enc4(enc3_s), self.enc4(enc3_q)
        enc5_s, enc5_q = self.enc5(enc4_s), self.enc5(enc4_q)

        # Cross-attention between query and support images
        attended_features = self.mechanism(enc5_q, enc5_s,support_masks)
        attended_features = attended_features.max(dim=0, keepdim=True)[0]  # (N, C, H, W)

        # Fusion step
        fusion = torch.cat([enc5_q * attended_features], dim=1)

        # Decoder with skip connections
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
        up1 = torch.cat([up1, enc1_q], dim=1)  # Use original enc1_q directly
        up1 = self.dec1(up1)

        # Final segmentation output (no extra interpolation)
        return self.final_conv(up1)    
    
    
def get_model(model_name='resnet50', mechanism='cross_attention', weight_path=None):
    """
    Creates and returns a Siamese U-Net model with the specified backbone and mechanism.
    
    Args:
        model_name (str): Backbone model ('resnet50' or 'vgg16').
        mechanism (str): Feature fusion mechanism ('cross_attention', etc.).
        weight_path (str, optional): Path to pre-trained weights. Default is None.
    
    Returns:
        nn.Module: The initialized model.
    
    Raises:
        ValueError: If an unsupported model_name is provided.
    """
    model_classes = {
        'resnet50': SiameseUNetResNet50,
        'vgg16': SiameseUNetVGG16
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_classes.keys())}.")

    model = model_classes[model_name](out_channels=1, mechanism=mechanism)

    if weight_path:
        load_model_weights(model, weight_path)

    return model

def load_model_weights(model, checkpoint_path='model_weights/best_model.pth'):
    """
    Load weights into a given model from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model instance to load weights into.
        checkpoint_path (str): Path to the saved model weights (.pth file).
    """
    try:
        # Load the saved weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        print(f"✅ Model weights loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")