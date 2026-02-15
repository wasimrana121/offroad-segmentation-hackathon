"""
HIGH-PERFORMANCE SEMANTIC SEGMENTATION TRAINING
Target: IoU 0.75+
Optimized for Google Colab / RTX GPUs

Features:
- DINOv2 ViT-Small backbone (frozen)
- Advanced segmentation head with multi-scale features
- Combined loss: Focal + Dice + Boundary
- Progressive training with multi-resolution
- Strong augmentation pipeline
- Test-Time Augmentation
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import json
import gc
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths (modify these for your setup)
    DATA_ROOT = '/content/drive/MyDrive/iba/Offroad_Segmentation_Training_Dataset'
    OUTPUT_DIR = '/content/drive/MyDrive/iba/training_results_final'
    
    # Model
    BACKBONE = "dinov2_vits14"  # Small ViT backbone
    HIDDEN_DIM = 256  # Increased from 192
    NUM_CLASSES = 11
    
    # Training - Progressive resolution
    IMAGE_SIZE = (392, 700)  # (H, W) - Higher resolution for better results
    BATCH_SIZE = 4  # Adjust based on GPU
    
    # Optimization
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 60
    WARMUP_EPOCHS = 3
    
    # Early stopping
    PATIENCE = 25
    MIN_DELTA = 0.001
    
    # Loss weights
    FOCAL_WEIGHT = 1.0
    DICE_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 0.5
    
    # Other
    SEED = 42
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Class mapping
    VALUE_MAP = {
        0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
        550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
    }

config = Config()

# ============================================================================
# SETUP
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(config.SEED)

# Auto-detect GPU and set batch size
def get_gpu_info():
    if not torch.cuda.is_available():
        return "CPU", 0, 2
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Auto batch size
    if "A100" in device_name or total_memory > 40:
        batch_size = 8
    elif "V100" in device_name or total_memory > 15:
        batch_size = 6
    elif "T4" in device_name or total_memory > 12:
        batch_size = 5
    else:
        batch_size = 4
    
    return device_name, total_memory, batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name, gpu_memory, auto_batch_size = get_gpu_info()

# Override batch size if auto-detected
config.BATCH_SIZE = auto_batch_size

print("=" * 80)
print("HIGH-PERFORMANCE SEGMENTATION TRAINING - TARGET IoU 0.75+")
print("=" * 80)
print(f"Device: {device}")
print(f"GPU: {gpu_name}")
print(f"Memory: {gpu_memory:.1f} GB")
print(f"Batch Size: {config.BATCH_SIZE} (auto-detected)")
print(f"Image Size: {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}")
print("=" * 80)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_train_augmentation(h, w, phase='strong'):
    """
    Multi-phase augmentation strategy
    phase: 'strong' for early training, 'medium' for fine-tuning
    """
    if phase == 'strong':
        return A.Compose([
            A.Resize(h, w),
            
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ),
            
            # Photometric
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1
                ),
            ], p=0.5),
            
            # Weather/Environmental
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=1
                ),
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.1,
                    p=1
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    src_radius=100,
                    p=1
                ),
            ], p=0.3),
            
            # Quality degradation
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ], p=0.2),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=1),
                A.GaussianBlur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.2),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    else:  # medium phase
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def get_val_augmentation(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# ============================================================================
# DATASET
# ============================================================================

def convert_mask(mask, value_map):
    """Convert raw mask values to class IDs"""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, value_map=None):
        self.image_dir = os.path.join(root_dir, split, 'Color_Images')
        self.mask_dir = os.path.join(root_dir, split, 'Segmentation')
        self.transform = transform
        self.value_map = value_map
        self.image_ids = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, img_id)
        mask = Image.open(mask_path)
        mask = convert_mask(mask, self.value_map)
        mask = np.array(mask)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask is long tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AdvancedSegHead(nn.Module):
    """
    Advanced segmentation head with:
    - ASPP for multi-scale features
    - CBAM attention
    - Skip connections
    - Boundary refinement
    """
    def __init__(self, in_channels, num_classes, tokenH, tokenW, hidden_dim=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ASPP Module
        self.aspp1 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=1)
        self.aspp2 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=2)
        self.aspp3 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=3)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            CBAM(hidden_dim),
            nn.Dropout2d(0.1)
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # Boundary head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x):
        # x: [B, N, C] from backbone
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        # Project
        x = self.input_proj(x)
        
        # ASPP
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_pool(x)
        x4 = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate ASPP features
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Fusion
        x = self.fusion(x)
        
        # Refinement
        x = x + self.refine(x)
        
        # Predictions
        seg = self.classifier(x)
        boundary = self.boundary_head(x)
        
        return seg, boundary


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for better boundary segmentation"""
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary loss to improve edge accuracy"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def get_boundary(self, mask):
        """Extract boundaries from segmentation mask"""
        # Sobel operator
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        kernel_x = kernel_x.view(1, 1, 3, 3).to(mask.device)
        kernel_y = kernel_y.view(1, 1, 3, 3).to(mask.device)
        
        mask_float = mask.float().unsqueeze(1)
        edge_x = F.conv2d(mask_float, kernel_x, padding=1)
        edge_y = F.conv2d(mask_float, kernel_y, padding=1)
        
        boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        boundary = (boundary > 0).float().squeeze(1)
        
        return boundary
    
    def forward(self, pred_boundary, target_mask):
        target_boundary = self.get_boundary(target_mask)
        return self.bce(pred_boundary.squeeze(1), target_boundary)


class CombinedLoss(nn.Module):
    """Combined loss function"""
    def __init__(self, num_classes, focal_weight=1.0, dice_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.boundary_loss = BoundaryLoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, pred_seg, pred_boundary, target):
        focal = self.focal_loss(pred_seg, target)
        dice = self.dice_loss(pred_seg, target)
        boundary = self.boundary_loss(pred_boundary, target)
        
        total = (self.focal_weight * focal + 
                self.dice_weight * dice + 
                self.boundary_weight * boundary)
        
        return total, {'focal': focal.item(), 'dice': dice.item(), 'boundary': boundary.item()}


# ============================================================================
# METRICS
# ============================================================================

@torch.no_grad()
def compute_iou(pred, target, num_classes=11):
    """Compute mean IoU"""
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    
    iou_per_class = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().item())
    
    return np.nanmean(iou_per_class)


@torch.no_grad()
def evaluate_model(model, backbone, dataloader, criterion, device, use_tta=True):
    """Evaluate model with optional TTA"""
    model.eval()
    
    total_loss = 0
    iou_scores = []
    loss_components = defaultdict(float)
    
    for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        with autocast():
            # Backbone features
            features = backbone.forward_features(images)["x_norm_patchtokens"]
            
            # Forward pass
            seg_logits, boundary_logits = model(features)
            seg_logits = F.interpolate(seg_logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            boundary_logits = F.interpolate(boundary_logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            # TTA: Horizontal flip
            if use_tta:
                images_flip = torch.flip(images, dims=[3])
                features_flip = backbone.forward_features(images_flip)["x_norm_patchtokens"]
                seg_logits_flip, boundary_logits_flip = model(features_flip)
                seg_logits_flip = F.interpolate(seg_logits_flip, size=masks.shape[1:], mode='bilinear', align_corners=False)
                seg_logits_flip = torch.flip(seg_logits_flip, dims=[3])
                
                seg_logits = (seg_logits + seg_logits_flip) / 2
            
            # Loss
            loss, components = criterion(seg_logits, boundary_logits, masks)
            
            # Metrics
            iou = compute_iou(seg_logits, masks, config.NUM_CLASSES)
        
        total_loss += loss.item()
        iou_scores.append(iou)
        for k, v in components.items():
            loss_components[k] += v
    
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_iou = np.mean(iou_scores)
    avg_components = {k: v / n for k, v in loss_components.items()}
    
    return avg_loss, avg_iou, avg_components


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, backbone, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    iou_scores = []
    loss_components = defaultdict(float)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Backbone features (frozen)
        with torch.no_grad():
            features = backbone.forward_features(images)["x_norm_patchtokens"]
        
        # Forward pass with AMP
        with autocast():
            seg_logits, boundary_logits = model(features)
            seg_logits = F.interpolate(seg_logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            boundary_logits = F.interpolate(boundary_logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            loss, components = criterion(seg_logits, boundary_logits, masks)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            iou = compute_iou(seg_logits, masks, config.NUM_CLASSES)
        
        total_loss += loss.item()
        iou_scores.append(iou)
        for k, v in components.items():
            loss_components[k] += v
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })
    
    n = len(dataloader)
    avg_loss = total_loss / n
    avg_iou = np.mean(iou_scores)
    avg_components = {k: v / n for k, v in loss_components.items()}
    
    return avg_loss, avg_iou, avg_components


def save_checkpoint(model, optimizer, epoch, val_iou, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'config': vars(config)
    }, filepath)


def plot_training_history(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_iou'], label='Val', linewidth=2)
    axes[0, 1].axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target (0.75)')
    axes[0, 1].set_title('Mean IoU', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    axes[0, 2].plot(history['lr'], linewidth=2, color='green')
    axes[0, 2].set_title('Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('LR')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(alpha=0.3)
    
    # Loss components
    axes[1, 0].plot(history['focal_loss'], label='Focal', linewidth=2)
    axes[1, 0].plot(history['dice_loss'], label='Dice', linewidth=2)
    axes[1, 0].plot(history['boundary_loss'], label='Boundary', linewidth=2)
    axes[1, 0].set_title('Loss Components', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # GPU Memory
    if 'gpu_memory' in history:
        axes[1, 1].plot(history['gpu_memory'], linewidth=2, color='red')
        axes[1, 1].set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].grid(alpha=0.3)
    
    # Best metrics text
    best_val_iou = max(history['val_iou'])
    best_epoch = history['val_iou'].index(best_val_iou) + 1
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.7, 'Best Results', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
    axes[1, 2].text(0.5, 0.5, f'Val IoU: {best_val_iou:.4f}', 
                    ha='center', va='center', fontsize=14)
    axes[1, 2].text(0.5, 0.3, f'Epoch: {best_epoch}', 
                    ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Display in Colab
    try:
        from IPython.display import Image, display
        display(Image(save_path))
    except:
        pass


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Create datasets
    h, w = config.IMAGE_SIZE
    
    train_dataset = SegmentationDataset(
        config.DATA_ROOT,
        split='train',
        transform=get_train_augmentation(h, w, phase='strong'),
        value_map=config.VALUE_MAP
    )
    
    val_dataset = SegmentationDataset(
        config.DATA_ROOT,
        split='val',
        transform=get_val_augmentation(h, w),
        value_map=config.VALUE_MAP
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Load backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', config.BACKBONE)
    backbone.eval()
    backbone.to(device)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    print("Backbone loaded and frozen")
    
    # Get embedding dimension
    with torch.no_grad():
        sample_input = torch.randn(1, 3, h, w).to(device)
        sample_output = backbone.forward_features(sample_input)["x_norm_patchtokens"]
        embedding_dim = sample_output.shape[2]
        tokenH = h // 14
        tokenW = w // 14
    
    print(f"Embedding dim: {embedding_dim}")
    print(f"Token grid: {tokenH}x{tokenW}")
    
    del sample_input, sample_output
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create model
    model = AdvancedSegHead(
        in_channels=embedding_dim,
        num_classes=config.NUM_CLASSES,
        tokenH=tokenH,
        tokenW=tokenW,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        config.NUM_CLASSES,
        focal_weight=config.FOCAL_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        boundary_weight=config.BOUNDARY_WEIGHT
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'focal_loss': [], 'dice_loss': [], 'boundary_loss': [],
        'lr': [], 'gpu_memory': []
    }
    
    best_val_iou = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    for epoch in range(1, config.MAX_EPOCHS + 1):
        # Train
        train_loss, train_iou, train_components = train_epoch(
            model, backbone, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_iou, val_components = evaluate_model(
            model, backbone, val_loader, criterion, device, use_tta=True
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # GPU memory
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['focal_loss'].append(val_components['focal'])
        history['dice_loss'].append(val_components['dice'])
        history['boundary_loss'].append(val_components['boundary'])
        history['lr'].append(current_lr)
        history['gpu_memory'].append(gpu_mem)
        
        # Print summary
        print(f"\nEpoch {epoch}/{config.MAX_EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.6f} | GPU: {gpu_mem:.2f}GB")
        
        # Save best model
        if val_iou > best_val_iou + config.MIN_DELTA:
            best_val_iou = val_iou
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_iou,
                os.path.join(config.OUTPUT_DIR, 'best_model.pth')
            )
            
            print(f"  ✓ New best model! IoU: {val_iou:.4f}")
            
            if val_iou >= 0.75:
                print(f"  🎯 TARGET REACHED! IoU: {val_iou:.4f} >= 0.75")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏸ Early stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_iou,
                os.path.join(config.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_iou,
        os.path.join(config.OUTPUT_DIR, 'final_model.pth')
    )
    
    # Plot training curves
    plot_training_history(
        history,
        os.path.join(config.OUTPUT_DIR, 'training_curves.png')
    )
    
    # Save history
    with open(os.path.join(config.OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    
    if best_val_iou >= 0.75:
        print("\n🎉 CONGRATULATIONS! Target IoU of 0.75+ achieved!")
    else:
        print(f"\n📊 Best IoU: {best_val_iou:.4f} / 0.75")
        print("Consider: longer training, higher resolution, or unfreezing backbone")


if __name__ == "__main__":
    main()