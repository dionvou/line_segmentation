import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image, ImageOps
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training




def train_model(model, train_loader, valid_loader, criterion, epochs=10, lr=1e-4, patience=10, path='model_weights/best_model.pth'):
    """
    Trains a given model using a specified dataset and loss function.
    Implements early stopping, AdamW as optimizer and a cosine annealing learning rate scheduler.
    Uses mixed precision for memory efficiency.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device)  # ✅ Works for older PyTorch versions


    best_val_loss = float('inf')
    stopping_counter = 0
    best_model_state = None  # Store the best model state

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for support_imgs, support_masks, query_img, query_mask in train_loader:
            support_imgs = torch.cat(support_imgs).to(device, dtype=torch.float16)  # Reduce precision
            support_masks = torch.cat(support_masks).to(device, dtype=torch.float16)
            query_img = query_img.to(device, dtype=torch.float16)
            query_mask = query_mask.to(device, dtype=torch.float16)
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):  # Enables mixed precision for efficiency
                query_pred = model(support_imgs, query_img, support_masks)
                loss = criterion(query_pred.squeeze(1), query_mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients
            scaler.step(optimizer)
            scaler.update()  # Update the scaler for next step
            
            total_loss += loss.detach().item()  # Use .detach() to free memory
        
        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for support_imgs, support_masks, query_img, query_mask in valid_loader:
                support_imgs = torch.cat(support_imgs).to(device, dtype=torch.float16)
                support_masks = torch.cat(support_masks).to(device, dtype=torch.float16)
                query_img = query_img.to(device, dtype=torch.float16)
                query_mask = query_mask.to(device, dtype=torch.float16)

                with torch.amp.autocast(device_type="cuda"):
                    query_pred = model(support_imgs, query_img, support_masks)
                    loss = criterion(query_pred.squeeze(1), query_mask)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stopping_counter = 0
            print('Saving model')
            best_model_state = model.state_dict().copy()  # Save best model state
            torch.save(model.state_dict(), path)
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_model_state is not None:
        print("Restoring best model state.")
        model.load_state_dict(best_model_state)


def get_predictions(model, dataloader):
    """
    Generates predictions using a trained model with reduced memory usage.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  

    predictions = []
    
    with torch.no_grad():
        for support_imgs, support_masks, query_img, _ in dataloader:
            support_imgs = torch.cat(support_imgs).to(device, dtype=torch.float16)
            support_masks = torch.cat(support_masks).to(device, dtype=torch.float16)
            query_img = query_img.to(device, dtype=torch.float16)

            with torch.amp.autocast(device_type="cuda"):
                query_pred = model(support_imgs, query_img, support_masks)
                query_pred = torch.sigmoid(query_pred.squeeze(1))

            query_pred = query_pred.cpu().half().numpy()  # Convert to float16 before moving to CPU
            predictions.extend(query_pred)

    return predictions   

import torch

# def load_model_weights(model, checkpoint_path='model_weights/best_model.pth', device="cpu"):
#     """
#     Load weights into a given model from a checkpoint file.
    
#     Args:
#         model (torch.nn.Module): The model instance to load weights into.
#         checkpoint_path (str): Path to the saved model weights (.pth file).
#         device (str): Device to load the model on ("cpu" or "cuda").
#     """
#     try:
#         # Load the saved weights
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
#         model.to(device)  # Move model to the specified device
#         print(f"✅ Model weights loaded successfully from {checkpoint_path}")
#     except Exception as e:
#         print(f"❌ Error loading model weights: {e}")
    

def save_images(images, save_folder='images/'):
    for idx, image in enumerate(images):
        if save_folder is not None:
            save_folder = save_folder
            image = (image).astype(np.uint8)
            image_save_path = os.path.join(save_folder, f"image_{idx + 1}.png")
            Image.fromarray(image).save(image_save_path)
     
# def train_model(model, train_loader, valid_loader, criterion, epochs=10, lr=1e-4, patience=10, path='model_weights/best_model.pth'):
#     """
#     Trains a given model using a specified dataset and loss function.
#     Implements early stopping, AdamW as optimizer and a cosine annealing learning rate scheduler.
    
#     Args:
#         model (torch.nn.Module): The model to be trained.
#         train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
#         valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
#         criterion (torch.nn.Module): Loss function.
#         epochs (int, optional): Number of training epochs. Default is 10.
#         lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
#         patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 10.
#         path (str, optional): Path to save the best model. Default is 'best_model.pth'.
    
#     Returns:
#         None
#     """
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
#     best_val_loss = float('inf')
#     stopping_counter = 0
#     best_model_state = None  # Store the best model state
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
        
#         for support_imgs, support_masks, query_img, query_mask in train_loader:
#             support_imgs = torch.cat(support_imgs).to(device)
#             support_masks = torch.cat(support_masks).to(device)
#             query_img = query_img.to(device)
#             query_mask = query_mask.to(device)
            
#             optimizer.zero_grad()
#             query_pred = model(support_imgs, query_img,support_masks)
#             loss = criterion(query_pred.squeeze(1), query_mask)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         avg_train_loss = total_loss / len(train_loader)
        
#         scheduler.step()
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for support_imgs, support_masks, query_img, query_mask in valid_loader:
#                 support_imgs = torch.cat(support_imgs).to(device)
#                 support_masks = torch.cat(support_masks).to(device)
#                 query_img = query_img.to(device)
#                 query_mask = query_mask.to(device)
                
#                 query_pred = model(support_imgs, query_img, support_masks)
#                 loss = criterion(query_pred.squeeze(1), query_mask)
#                 val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(valid_loader)
#         print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             stopping_counter = 0
#             print('Saving model')
#             best_model_state = model.state_dict().copy()  # Save best model state
#             torch.save(model.state_dict(), path)
#         else:
#             stopping_counter += 1
#             if stopping_counter >= patience:
#                 print(f"Early stopping triggered after {epoch+1} epochs.")
#                 break
#     if best_model_state is not None:
#         print("Restoring best model state.")
#         model.load_state_dict(best_model_state)
    

# def get_predictions(model, dataloader):
#     """
#     Generates predictions using a trained model.

#     Args:
#         model (torch.nn.Module): The trained model.
#         dataloader (torch.utils.data.DataLoader): DataLoader containing the validation or test dataset.
#         device (str, optional): Device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.

#     Returns:
#         list of numpy arrays: List of predicted query masks as NumPy arrays.
#     """

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()  # Set the model to evaluation mode
    
#     predictions = []
    
#     with torch.no_grad():  # Disable gradient computation
#         for support_imgs, support_masks, query_img, _ in dataloader:
#             # Move inputs to the specified device
#             support_imgs = torch.cat(support_imgs).to(device)
#             support_masks = torch.cat(support_masks).to(device)
#             query_img = query_img.to(device)
            
#             # Get model prediction
#             query_pred = model(support_imgs, query_img,support_masks)
#             query_pred = torch.sigmoid(query_pred.squeeze(1))  # Apply sigmoid activation
            
#             # Convert to NumPy array and store
#             query_pred = query_pred.cpu().numpy()  # Move to CPU before converting
#             predictions.extend(query_pred)

#     return predictions  # List of predicted query masks


def reconstruct(dataloader, tile_list, save_folder=None):
    """
    Reconstructs full-sized images from a list of predicted tiles.

    This function reassembles original images by placing tiles in their respective positions, 
    averaging overlapping regions, and removing any padding added during the tiling process.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset with image file paths and tiling parameters.
        tile_list (list of numpy arrays): List of predicted tiles corresponding to the images in the dataset.
                save_folder (str, optional): Folder path where reconstructed images will be saved. 
                                     If None, images will not be saved. Default is None.

    Returns:
        list of numpy arrays: Reconstructed images, each restored to its original size.

    Notes:
        - Assumes tiles are provided in the same order as they were extracted.
        - Overlapping tile areas are averaged to ensure smooth reconstruction.
        - Any padding added during tiling is removed to recover the original dimensions.
    """
    # Create save directory if it doesn't exist
    dataset = dataloader.dataset
    tile_list = [np.array(tile) if not isinstance(tile, np.ndarray) else tile for tile in tile_list]
    reconstructed_images = []
    tile_list_idx = 0
    # For each image in the dataset
    for image_idx, image_path in enumerate(dataset.image_files):

        original_image = Image.open(image_path)
        orig_width, orig_height = original_image.size

        # Compute padding to match dataset tiling
        pad_w = (dataset.stride - (orig_width % dataset.stride)) % dataset.stride
        pad_h = (dataset.stride - (orig_height % dataset.stride)) % dataset.stride

        padded_width = orig_width + pad_w
        padded_height = orig_height + pad_h

        # Create empty canvas for reconstruction
        reconstructed_image = np.zeros((padded_height, padded_width), dtype=np.float32)
        overlap_count = np.zeros((padded_height, padded_width), dtype=np.int32)

        # Calculate total times of current image
        tiles_in_height = (padded_height - dataset.tile_size[0]) // dataset.stride + 1
        tiles_in_width = (padded_width - dataset.tile_size[1]) // dataset.stride + 1
        total_tiles = tiles_in_height * tiles_in_width

        # Process batches from DataLoader
        tile_idx = 0  # Track tile index within the image

        while tile_idx < total_tiles:
            image_tile = tile_list[tile_list_idx]

            # Compute tile placement indices
            row_idx = tile_idx // ((padded_width - dataset.tile_size[1]) // dataset.stride + 1)
            col_idx = tile_idx % ((padded_width - dataset.tile_size[1]) // dataset.stride + 1)

            top_left_y = row_idx * dataset.stride
            top_left_x = col_idx * dataset.stride

            # Add tile values to the corresponding region
            reconstructed_image[top_left_y:top_left_y + dataset.tile_size[0],
                                top_left_x:top_left_x + dataset.tile_size[1]] += image_tile
            overlap_count[top_left_y:top_left_y + dataset.tile_size[0],
                          top_left_x:top_left_x + dataset.tile_size[1]] += 1

            tile_idx += 1  # Increment tile index
            tile_list_idx +=1

        # Normalize overlapping areas
        overlap_count[overlap_count == 0] = 1
        reconstructed_image /= overlap_count

        # Remove padding (crop back to original size)
        reconstructed_image = reconstructed_image[:orig_height, :orig_width]
        reconstructed_images.append(reconstructed_image)

        # Convert to uint8 and save
        if save_folder is not None:
            save_folder = save_folder
            reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
            image_save_path = os.path.join(save_folder, f"reconstructed_{image_idx + 1}.png")
            Image.fromarray(reconstructed_image).save(image_save_path)

    return reconstructed_images

def thresholding_images(image_list, th):
    """
    Applies thresholding to a list of images.

    Args:
        image_list (list of numpy arrays): List of grayscale images (NumPy arrays).
        th (float): Threshold value. Pixels greater than this value are set to 1, others to 0.

    Returns:
        list of numpy arrays: List of thresholded images with binary values (0 or 1).

    Notes:
        - Assumes input images are in the range [0, 1] or [0, 255]. If in [0, 255], normalization may be required before thresholding.
        - Modifies images in-place; if preserving originals is needed, consider copying before thresholding.
    """
    for img in image_list:
        img[img > th] = 1
        img[img <= th] = 0
    return image_list

def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological erosion to the image.

    Parameters:
        image (np.ndarray): Input binary image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times erosion is applied.

    Returns:
        np.ndarray: Eroded image.
    """
    # Create a square structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded = cv2.erode(image, kernel, iterations=iterations)

    return eroded

def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological dilation to the image.

    Parameters:
        image (np.ndarray): Input binary image.
        kernel_size (int): Size of the structuring element.
        iterations (int): Number of times dilation is applied.

    Returns:
        np.ndarray: Dilated image.
    """
    # Create a square structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated = cv2.dilate(image, kernel, iterations=iterations)

    return dilated

def remove_small_objects(image: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Removes small objects (noise) from a binary image using contour filtering.

    Parameters:
        image (np.ndarray): Input binary image.
        min_size (int): Minimum area of connected components to keep.

    Returns:
        np.ndarray: Processed image with small objects removed (black).
    """
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Normalize if needed

    # Threshold to ensure binary format (0 and 255)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the binary image
    mask = np.copy(binary)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size:  # Remove small objects
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)  # Fill with black (0)

    return mask

def intersection_over_union(tp, fp, fn):
    return tp / (tp + fp + fn + 1e-8)

def evaluate_IU(dataloader, predictions, th=0.5):
    """
    Evaluates Intersection over Union (IoU) at both pixel and line levels for a given dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset with ground truth labels.
        predictions (list of numpy arrays): List of predicted masks corresponding to dataset images.
        th (float, optional): Threshold to binarize predicted masks. Default is 0.5.

    Returns:
        dict: A dictionary containing:
            - "Pixel IU" (float): Mean Intersection over Union at pixel level.
            - "Line IU" (float): Mean Intersection over Union at line level.
            - "Pixel Accuracy" (float): Overall pixel-wise accuracy.

    Notes:
        - Pixel IU is computed as IoU between ground truth and predicted pixels.
        - Line IU measures IoU between extracted connected components of ground truth and predictions.
        - A prediction component is considered a match if it has at least 75% overlap with a ground truth component.
    """
    dataset = dataloader.dataset
    pixel_IUs = []
    line_IUs = []
    total_pixels = 0
    correct_pixels = 0

    for file, pred_mask in zip(dataset.label_files, predictions):
        print(file)
        # Load ground truth label
        label = Image.open(file).convert('L')
        label = np.array(label, dtype=np.uint8) / 255
        label = label > 0.5  # Convert to binary mask

        # Convert prediction to binary mask
        pred_mask = np.array(pred_mask > th, dtype=np.uint8)

        # Pixel-level IU
        tp = np.sum((pred_mask == 1) & (label == 1))
        fp = np.sum((pred_mask == 1) & (label == 0))
        fn = np.sum((pred_mask == 0) & (label == 1))

        pixel_IU = intersection_over_union(tp, fp, fn)
        pixel_IUs.append(pixel_IU)

        # Compute line-level IU
        label_components, pred_components = extract_connected_components(label, pred_mask)
        matched_lines = 0
        total_gt_lines = len(label_components)
        total_pred_lines = len(pred_components)
        print('Number of componets in label:',total_gt_lines)
        print('Number of componets in pred:',total_pred_lines)

        for pred_comp in pred_components:
            best_match = max([match_score(pred_comp, gt_comp) for gt_comp in label_components], default=0)
            if best_match >= 0.75:
                matched_lines += 1

        line_IU = matched_lines / (total_gt_lines + total_pred_lines - matched_lines + 1e-8)
        line_IUs.append(line_IU)

        print('Matched_lines', matched_lines)

        # Pixel Accuracy
        total_pixels += label.size
        correct_pixels += np.sum(pred_mask == label)

    return {
        "Pixel IU": np.mean(pixel_IUs),
        "Line IU": np.mean(line_IUs),
        "Pixel Accuracy": correct_pixels / total_pixels
    }

def extract_connected_components(label, pred):
    """
    Extract connected components from label (ground truth) and pred (predicted mask).

    Args:
        label (numpy.ndarray): Binary ground truth mask.
        pred (numpy.ndarray): Binary predicted mask.

    Returns:
        label_components (list of lists): List of connected components in ground truth.
        pred_components (list of lists): List of connected components in predictions.
    """

    def get_components(binary_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        components = []

        for i in range(1, num_labels):  # Ignore background label (0)
            component = np.column_stack(np.where(labels == i))  # Get coordinates
            components.append(component.tolist())

        return components

    label_components = get_components(label)
    pred_components = get_components(pred)

    return label_components, pred_components

def match_score(pred_component, gt_component):
    # Compute match score based on Intersection over Union
    pred_set = set(map(tuple, pred_component))
    gt_set = set(map(tuple, gt_component))
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / (union + 1e-8)


def dice_loss(pred, target, weight_pos=1.0, weight_neg=1.0, epsilon=1e-6):
    """Compute weighted Dice Loss for binary segmentation."""
    pred_prob = torch.sigmoid(pred)  # Convert logits to probabilities

    # Flatten tensors
    pred_flat = pred_prob.view(-1)
    target_flat = target.view(-1)

    # Compute intersection and union
    intersection = torch.sum(weight_pos * pred_flat * target_flat)
    union = torch.sum(weight_pos * pred_flat) + torch.sum(weight_neg * target_flat)

    # Compute Dice coefficient
    dice_coeff = (2. * intersection + epsilon) / (union + epsilon)

    # Return Dice loss
    return 1 - dice_coeff

def focal_loss(pred, target, alpha=0.25, gamma=2.0, epsilon=1e-6):
    """Compute Focal Loss for binary segmentation."""
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred)

    # Compute p_t (probability of target class)
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)

    # Compute focal weight
    focal_weight = (1 - p_t) ** gamma

    # Apply alpha balancing
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)

    # Compute final loss
    loss = alpha_factor * focal_weight * bce_loss

    return loss.mean()
