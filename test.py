"""
SEGMENTATION TESTING & INFERENCE
Matches training code architecture exactly.

Modes:
  1. evaluate   — full val/test set metrics (IoU, per-class breakdown)
  2. predict    — run inference on a single image or folder
  3. visualize  — save colour-coded predictions side-by-side with GT
  4. confusion  — confusion matrix + per-class IoU table

Usage (Colab / terminal):
  # Full evaluation on val set
  python test_segmentation.py --mode evaluate --split val

  # Single image prediction
  python test_segmentation.py --mode predict --input /path/to/image.jpg

  # Predict all images in a folder
  python test_segmentation.py --mode predict --input /path/to/folder/

  # Visual comparison (needs GT masks)
  python test_segmentation.py --mode visualize --split val --num_samples 10

  # Confusion matrix
  python test_segmentation.py --mode confusion --split val
"""

import argparse
import os
import json
import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# CONFIGURATION  (must match training Config exactly)
# ============================================================================

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    # Your public test set root — contains Color_Images/ directly (no split subfolder)
    DATA_ROOT    = '/content/drive/MyDrive/iba/test_public_80'
    OUTPUT_DIR   = '/content/drive/MyDrive/iba/test_results'
    MODEL_PATH   = '/content/drive/MyDrive/iba/best_model.pth'

    # ── Set this based on your test set structure ───────────────────────────
    # Layout A — FLAT WITH GT (your current test set):
    #   test_public_80/Color_Images/   ← images here
    #   test_public_80/Segmentation/   ← masks here
    #   → set NO_GT=False, FLAT_WITH_GT=True
    #
    # Layout B — FLAT NO GT (public leaderboard, no masks):
    #   test_public_80/Color_Images/
    #   → set NO_GT=True, FLAT_WITH_GT=True
    #
    # Layout C — SPLIT SUBFOLDERS (training/val dataset):
    #   test_public_80/{split}/Color_Images/
    #   test_public_80/{split}/Segmentation/
    #   → set NO_GT=False, FLAT_WITH_GT=False
    NO_GT        = False   # ← False because you HAVE Segmentation/ folder
    FLAT_WITH_GT = True    # ← True because there is NO split subfolder

    BACKBONE     = "dinov2_vits14"
    HIDDEN_DIM   = 256
    NUM_CLASSES  = 11
    IMAGE_SIZE   = (392, 700)       # (H, W) — must match training

    # Test-Time Augmentation
    USE_TTA      = True             # horizontal flip TTA
    TTA_SCALES   = [1.0]            # add 0.75, 1.25 for multi-scale TTA

    BATCH_SIZE   = 4
    NUM_WORKERS  = 2

    CLASS_NAMES  = [
        'background', 'class_100', 'class_200', 'class_300', 'class_500',
        'class_550',  'class_600', 'class_700', 'class_800', 'class_7100',
        'class_10000'
    ]

    # Colour palette for visualisation (BGR → RGB already converted below)
    CLASS_COLORS = [
        (0,   0,   0),    # 0  background  — black
        (128, 0,   0),    # 1  class_100   — dark red
        (0,   128, 0),    # 2  class_200   — dark green
        (128, 128, 0),    # 3  class_300   — olive
        (0,   0,   128),  # 4  class_500   — dark blue
        (128, 0,   128),  # 5  class_550   — purple
        (0,   128, 128),  # 6  class_600   — teal
        (255, 128, 0),    # 7  class_700   — orange
        (255, 0,   128),  # 8  class_800   — pink
        (0,   255, 128),  # 9  class_7100  — lime
        (128, 255, 255),  # 10 class_10000 — light cyan
    ]

    VALUE_MAP = {
        0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
        550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
    }

config = Config()

# ============================================================================
# DEVICE
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# MODEL ARCHITECTURE  (copy-pasted from training — must stay identical)
# ============================================================================

class CBAM(nn.Module):
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
        return x * self.sigmoid(avg_out + max_out)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AdvancedSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, tokenH, tokenW, hidden_dim=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.aspp1 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=1)
        self.aspp2 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=2)
        self.aspp3 = ASPPModule(hidden_dim, hidden_dim // 4, dilation=3)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            CBAM(hidden_dim),
            nn.Dropout2d(0.1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier    = nn.Conv2d(hidden_dim, num_classes, 1)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x  = self.input_proj(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = F.interpolate(self.global_pool(x), size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        x  = self.fusion(torch.cat([x1, x2, x3, x4], dim=1))
        x  = x + self.refine(x)
        return self.classifier(x), self.boundary_head(x)


# ============================================================================
# MODEL LOADER
# ============================================================================

def load_model(model_path, device):
    """Load backbone + seg head from checkpoint."""
    print(f"\nLoading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    h, w = config.IMAGE_SIZE
    tokenH, tokenW = h // 14, w // 14

    # Backbone
    backbone = torch.hub.load('facebookresearch/dinov2', config.BACKBONE)
    backbone.eval().to(device)

    # If checkpoint includes backbone state (from Phase 2 / boost script)
    if 'backbone_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        print("  Loaded fine-tuned backbone weights")

    for param in backbone.parameters():
        param.requires_grad = False

    # Infer embedding dim
    with torch.no_grad():
        sample = torch.randn(1, 3, h, w).to(device)
        embedding_dim = backbone.forward_features(sample)["x_norm_patchtokens"].shape[2]
    del sample; torch.cuda.empty_cache(); gc.collect()

    # Seg head
    model = AdvancedSegHead(
        in_channels=embedding_dim,
        num_classes=config.NUM_CLASSES,
        tokenH=tokenH, tokenW=tokenW,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    saved_iou   = checkpoint.get('val_iou', 'unknown')
    saved_epoch = checkpoint.get('epoch',   'unknown')
    print(f"  Epoch: {saved_epoch} | Saved val IoU: {saved_iou}")
    print(f"  Embedding dim: {embedding_dim} | Token grid: {tokenH}×{tokenW}")

    return backbone, model


# ============================================================================
# PREPROCESSING
# ============================================================================

def get_transform(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def preprocess_image(image_path, h, w):
    """Load and preprocess a single image. Returns tensor + original numpy."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()

    transform = get_transform(h, w)
    tensor = transform(image=image)['image']
    return tensor.unsqueeze(0), original   # [1,3,H,W], HxWx3


def convert_mask(mask, value_map):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


# ============================================================================
# INFERENCE  (single image, with TTA)
# ============================================================================

@torch.no_grad()
def predict_single(backbone, model, image_tensor, original_h, original_w):
    """
    Run inference on a single image tensor [1,3,H,W].
    Returns predicted class mask [H,W] in ORIGINAL image resolution.
    """
    h, w = config.IMAGE_SIZE
    image_tensor = image_tensor.to(device)

    with torch.amp.autocast('cuda'):
        features  = backbone.forward_features(image_tensor)["x_norm_patchtokens"]
        logits, _ = model(features)
        logits    = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

        if config.USE_TTA:
            # Horizontal flip TTA
            flipped     = torch.flip(image_tensor, dims=[3])
            feat_flip   = backbone.forward_features(flipped)["x_norm_patchtokens"]
            logits_flip, _ = model(feat_flip)
            logits_flip = F.interpolate(logits_flip, size=(h, w),
                                        mode='bilinear', align_corners=False)
            logits_flip = torch.flip(logits_flip, dims=[3])
            logits = (logits + logits_flip) / 2

        # Multi-scale TTA (if more than [1.0] in TTA_SCALES)
        for scale in config.TTA_SCALES:
            if scale == 1.0:
                continue
            sh, sw   = int(h * scale), int(w * scale)
            scaled   = F.interpolate(image_tensor, size=(sh, sw),
                                     mode='bilinear', align_corners=False)
            feat_s   = backbone.forward_features(scaled)["x_norm_patchtokens"]
            # Adjust token grid temporarily — we use a detached model call
            logits_s, _ = model(feat_s)   # note: this will only work if tokenH/W match
            logits_s = F.interpolate(logits_s, size=(h, w),
                                     mode='bilinear', align_corners=False)
            logits   = (logits + logits_s) / 2

    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize to original image resolution
    if (original_h, original_w) != (h, w):
        pred = cv2.resize(pred, (original_w, original_h),
                          interpolation=cv2.INTER_NEAREST)

    return pred


# ============================================================================
# COLOUR MASK RENDERING
# ============================================================================

def mask_to_color(mask):
    """Convert integer class mask → RGB colour image."""
    h, w    = mask.shape
    color   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(config.CLASS_COLORS):
        color[mask == cls_id] = rgb
    return color


def overlay(image, color_mask, alpha=0.5):
    """Blend colour mask over original image."""
    return (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)


# ============================================================================
# DATASET (for batch evaluation)
# ============================================================================

from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    """
    Supports two folder layouts:

    Layout A — flat (NO_GT=True, public test set, no ground-truth masks):
        DATA_ROOT/
          Color_Images/
            img001.png ...

    Layout B — split subfolders (NO_GT=False, training/val set, has GT masks):
        DATA_ROOT/
          {split}/
            Color_Images/
            Segmentation/
    """
    def __init__(self, root_dir, split='val', transform=None,
                 value_map=None, no_gt=False):
        self.no_gt     = no_gt
        self.transform = transform
        self.value_map = value_map

        flat = config.FLAT_WITH_GT or no_gt
        if no_gt and flat:
            # Layout B: flat, no GT
            self.image_dir = os.path.join(root_dir, 'Color_Images')
            self.mask_dir  = None
        elif flat:
            # Layout A: flat WITH GT  ← your test_public_80
            self.image_dir = os.path.join(root_dir, 'Color_Images')
            self.mask_dir  = os.path.join(root_dir, 'Segmentation')
        else:
            # Layout C: split subfolders  ← training dataset
            self.image_dir = os.path.join(root_dir, split, 'Color_Images')
            self.mask_dir  = os.path.join(root_dir, split, 'Segmentation')

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                f"Check DATA_ROOT and NO_GT setting in Config."
            )

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_ids = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"  Found {len(self.image_ids)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image  = cv2.imread(os.path.join(self.image_dir, img_id))
        if image is None:
            raise IOError(f"Cannot read image: {img_id}")
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.no_gt:
            # No ground-truth — return dummy zero mask so DataLoader shape is consistent
            h, w = config.IMAGE_SIZE
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = Image.open(os.path.join(self.mask_dir, img_id))
            mask = convert_mask(mask, self.value_map)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask, img_id


# ============================================================================
# METRICS HELPERS
# ============================================================================

def compute_iou_batch(preds, targets, num_classes):
    """
    preds  : [B, C, H, W] logits  OR  [B, H, W] class indices
    targets: [B, H, W] long
    Returns per-class IoU list and mean IoU (NaN-safe).
    """
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    preds   = preds.view(-1)
    targets = targets.view(-1)

    ious = []
    for c in range(num_classes):
        p = preds   == c
        t = targets == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return ious, float(np.nanmean(ious))


def compute_confusion_matrix(preds_all, targets_all, num_classes):
    """Accumulate confusion matrix across batches."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(preds_all, targets_all):
        p = pred.view(-1).cpu().numpy()
        t = target.view(-1).cpu().numpy()
        for true_cls in range(num_classes):
            mask = t == true_cls
            if mask.any():
                for pred_cls in range(num_classes):
                    cm[true_cls, pred_cls] += int((p[mask] == pred_cls).sum())
    return cm


# ============================================================================
# MODE 1 — EVALUATE
# ============================================================================

@torch.no_grad()
def run_evaluate(backbone, model, split='val'):
    if config.NO_GT:
        print("\n⚠️  NO_GT=True — test_public_80 has no ground-truth masks.")
        print("   Switching to predict mode (saving colour masks + overlays).")
        print("   Set NO_GT=False if your dataset DOES have a Segmentation/ folder.\n")
        run_predict(backbone, model,
                    input_path=os.path.join(config.DATA_ROOT, 'Color_Images'))
        return

    print(f"\n{'='*60}")
    print(f"EVALUATE  —  split: {split}")
    print(f"{'='*60}")

    h, w = config.IMAGE_SIZE
    dataset = SegmentationDataset(
        config.DATA_ROOT, split=split,
        transform=get_transform(h, w),
        value_map=config.VALUE_MAP,
        no_gt=config.NO_GT
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    total_iou     = []
    class_iou_acc = [[] for _ in range(config.NUM_CLASSES)]
    preds_all, targets_all = [], []

    for images, masks, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks  = masks.to(device)

        with torch.amp.autocast('cuda'):
            features  = backbone.forward_features(images)["x_norm_patchtokens"]
            logits, _ = model(features)
            logits    = F.interpolate(logits, size=masks.shape[1:],
                                      mode='bilinear', align_corners=False)

            if config.USE_TTA:
                flip_feat   = backbone.forward_features(
                                  torch.flip(images, dims=[3]))["x_norm_patchtokens"]
                logits_flip, _ = model(flip_feat)
                logits_flip = F.interpolate(logits_flip, size=masks.shape[1:],
                                            mode='bilinear', align_corners=False)
                logits_flip = torch.flip(logits_flip, dims=[3])
                logits = (logits + logits_flip) / 2

        cls_ious, mean_iou = compute_iou_batch(logits, masks, config.NUM_CLASSES)
        total_iou.append(mean_iou)
        for c, v in enumerate(cls_ious):
            if not np.isnan(v):
                class_iou_acc[c].append(v)

        preds_all.append(torch.argmax(logits, dim=1).cpu())
        targets_all.append(masks.cpu())

    # ── Results ──────────────────────────────────────────────────────────
    mean_iou = float(np.mean(total_iou))
    print(f"\nMean IoU  : {mean_iou:.4f}")
    print(f"\n{'Class':<15} {'IoU':>8}  {'Samples':>10}")
    print("-" * 38)

    for c, name in enumerate(config.CLASS_NAMES):
        vals = class_iou_acc[c]
        iou  = float(np.mean(vals)) if vals else float('nan')
        flag = " ✓" if iou >= 0.75 else (" ~" if iou >= 0.50 else " ✗")
        print(f"  {name:<13} {iou:>8.4f}  {len(vals):>10}{flag}")

    if mean_iou >= 0.75:
        print("\n🎯 TARGET IoU 0.75+ ACHIEVED!")
    elif mean_iou >= 0.70:
        print("\n🔥 Past 0.70 milestone!")
    else:
        print(f"\n📈 Current best: {mean_iou:.4f}")

    # Save JSON
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = {
        'split': split,
        'mean_iou': mean_iou,
        'per_class_iou': {
            config.CLASS_NAMES[c]: float(np.mean(class_iou_acc[c])) if class_iou_acc[c] else None
            for c in range(config.NUM_CLASSES)
        },
        'model_path': config.MODEL_PATH,
        'use_tta': config.USE_TTA,
    }
    out_path = os.path.join(config.OUTPUT_DIR, f'eval_{split}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return preds_all, targets_all, mean_iou


# ============================================================================
# MODE 2 — PREDICT  (single image or folder)
# ============================================================================

def run_predict(backbone, model, input_path):
    print(f"\n{'='*60}")
    print(f"PREDICT  —  input: {input_path}")
    print(f"{'='*60}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    h, w = config.IMAGE_SIZE

    # Collect image paths
    if os.path.isdir(input_path):
        exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        paths  = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"Found {len(paths)} images")
    else:
        paths = [input_path]

    for img_path in tqdm(paths, desc="Predicting"):
        tensor, original = preprocess_image(img_path, h, w)
        orig_h, orig_w   = original.shape[:2]

        pred_mask  = predict_single(backbone, model, tensor, orig_h, orig_w)
        color_mask = mask_to_color(pred_mask)
        blended    = overlay(original, color_mask, alpha=0.5)

        # Save outputs
        basename = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(
            os.path.join(config.OUTPUT_DIR, f'{basename}_mask.png'),
            cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(config.OUTPUT_DIR, f'{basename}_overlay.png'),
            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        )
        np.save(
            os.path.join(config.OUTPUT_DIR, f'{basename}_pred.npy'),
            pred_mask
        )

    print(f"\nSaved predictions → {config.OUTPUT_DIR}")


# ============================================================================
# MODE 3 — VISUALIZE  (side-by-side with GT)
# ============================================================================

@torch.no_grad()
def run_visualize(backbone, model, split='val', num_samples=8):
    print(f"\n{'='*60}")
    print(f"VISUALIZE  —  split: {split}  |  samples: {num_samples}")
    print(f"{'='*60}")

    h, w = config.IMAGE_SIZE

    if config.FLAT_WITH_GT or config.NO_GT:
        # Flat layout — Color_Images/ and Segmentation/ directly under DATA_ROOT
        image_dir = os.path.join(config.DATA_ROOT, 'Color_Images')
        mask_dir  = None if config.NO_GT else os.path.join(config.DATA_ROOT, 'Segmentation')
    else:
        # Split layout — DATA_ROOT/{split}/Color_Images + Segmentation
        image_dir = os.path.join(config.DATA_ROOT, split, 'Color_Images')
        mask_dir  = os.path.join(config.DATA_ROOT, split, 'Segmentation')

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_ids = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:num_samples]
    transform = get_transform(h, w)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Build legend patches
    patches = [
        mpatches.Patch(color=np.array(c) / 255.0, label=n)
        for n, c in zip(config.CLASS_NAMES, config.CLASS_COLORS)
    ]

    for img_id in tqdm(image_ids, desc="Visualizing"):
        # Load image
        image     = cv2.imread(os.path.join(image_dir, img_id))
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Predict
        transform_ = get_transform(h, w)
        tensor    = transform_(image=image)['image'].unsqueeze(0)
        pred_mask = predict_single(backbone, model, tensor, orig_h, orig_w)
        pred_color = mask_to_color(pred_mask)
        blended    = overlay(image, pred_color, alpha=0.5)

        if config.NO_GT:
            # 3-panel: original | prediction | overlay  (no GT available)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(img_id, fontsize=13, fontweight='bold')
            axes[0].imshow(image);      axes[0].set_title('Original',   fontweight='bold')
            axes[1].imshow(pred_color); axes[1].set_title('Prediction', fontweight='bold')
            axes[2].imshow(blended);    axes[2].set_title('Overlay',    fontweight='bold')
        else:
            # 4-panel: original | GT | prediction | overlay
            gt_mask = Image.open(os.path.join(mask_dir, img_id))
            gt_mask = convert_mask(gt_mask, config.VALUE_MAP)
            gt_color = mask_to_color(gt_mask)
            _, mean_iou = compute_iou_batch(
                torch.from_numpy(pred_mask).unsqueeze(0),
                torch.from_numpy(gt_mask).long().unsqueeze(0),
                config.NUM_CLASSES
            )
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            fig.suptitle(f'{img_id}  |  IoU = {mean_iou:.4f}',
                         fontsize=13, fontweight='bold')
            axes[0].imshow(image);      axes[0].set_title('Original',     fontweight='bold')
            axes[1].imshow(gt_color);   axes[1].set_title('Ground Truth', fontweight='bold')
            axes[2].imshow(pred_color); axes[2].set_title('Prediction',   fontweight='bold')
            axes[3].imshow(blended);    axes[3].set_title('Overlay',      fontweight='bold')

        for ax in axes:
            ax.axis('off')

        fig.legend(handles=patches, loc='lower center', ncol=6,
                   fontsize=8, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()

        basename = os.path.splitext(img_id)[0]
        save_path = os.path.join(config.OUTPUT_DIR, f'vis_{basename}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

        # Show in Colab
        try:
            from IPython.display import Image as IPImg, display
            display(IPImg(save_path))
        except:
            pass

    print(f"\nVisualisations saved → {config.OUTPUT_DIR}")


# ============================================================================
# MODE 4 — CONFUSION MATRIX
# ============================================================================

@torch.no_grad()
def run_confusion(backbone, model, split='val'):
    if config.NO_GT:
        print("\n⚠️  NO_GT=True — cannot build confusion matrix without ground-truth masks.")
        print("   Set NO_GT=False to use this mode.\n")
        return

    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX  —  split: {split}")
    print(f"{'='*60}")

    h, w = config.IMAGE_SIZE
    dataset = SegmentationDataset(
        config.DATA_ROOT, split=split,
        transform=get_transform(h, w),
        value_map=config.VALUE_MAP,
        no_gt=config.NO_GT
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

    for images, masks, _ in tqdm(loader, desc="Building confusion matrix"):
        images = images.to(device)
        masks  = masks.to(device)

        with torch.amp.autocast('cuda'):
            features  = backbone.forward_features(images)["x_norm_patchtokens"]
            logits, _ = model(features)
            logits    = F.interpolate(logits, size=masks.shape[1:],
                                      mode='bilinear', align_corners=False)

            if config.USE_TTA:
                ff = backbone.forward_features(
                         torch.flip(images, dims=[3]))["x_norm_patchtokens"]
                lf, _ = model(ff)
                lf = F.interpolate(lf, size=masks.shape[1:],
                                   mode='bilinear', align_corners=False)
                lf = torch.flip(lf, dims=[3])
                logits = (logits + lf) / 2

        preds = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
        tgts  = masks.view(-1).cpu().numpy()
        for t in range(config.NUM_CLASSES):
            m = tgts == t
            if m.any():
                for p in range(config.NUM_CLASSES):
                    cm[t, p] += int((preds[m] == p).sum())

    # ── Normalise & plot ──────────────────────────────────────────────────
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  /= row_sums

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(config.NUM_CLASSES)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True',      fontsize=11, fontweight='bold')
    ax.set_title(f'Normalised Confusion Matrix  ({split})', fontsize=13, fontweight='bold')

    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if row_sums[i, 0] > 0:
                val   = cm_norm[i, j]
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, f'confusion_{split}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Per-class precision / recall / IoU table ──────────────────────────
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'IoU':>8}")
    print("-" * 56)
    for c in range(config.NUM_CLASSES):
        tp  = cm[c, c]
        fp  = cm[:, c].sum() - tp
        fn  = cm[c, :].sum() - tp
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        f1    = 2 * prec * rec / (prec + rec + 1e-8)
        iou_c = tp / (tp + fp + fn + 1e-8)
        print(f"  {config.CLASS_NAMES[c]:<13} {prec:>10.4f} {rec:>10.4f} {f1:>8.4f} {iou_c:>8.4f}")

    print(f"\nConfusion matrix saved → {save_path}")

    try:
        from IPython.display import Image as IPImg, display
        display(IPImg(save_path))
    except:
        pass


# ============================================================================
# CLI
# ============================================================================

def is_notebook():
    """Detect if running inside Jupyter / Google Colab."""
    try:
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'Shell')   # Jupyter / Colab
    except NameError:
        return False                                        # plain Python script


def parse_args():
    """
    Parse args safely in both Colab notebooks and terminal scripts.
    In a notebook, argparse would try to parse Jupyter's kernel flags and crash.
    Instead we return a namespace built from Config defaults.
    """
    if is_notebook():
        # ── COLAB / JUPYTER: edit these values directly ──────────────────
        import argparse
        args = argparse.Namespace(
            # ── Edit these for your run ─────────────────────────────────
            # Modes: 'predict'   → save masks (works with NO_GT=True)
            #        'visualize' → visual panels (works with NO_GT=True, 3-panel)
            #        'evaluate'  → IoU metrics  (requires NO_GT=False + GT masks)
            #        'confusion' → confusion matrix (requires NO_GT=False)
            mode       = 'evaluate',          # ← GT available: evaluate/visualize/confusion all work
            model      = config.MODEL_PATH,
            split      = 'val',               # only used when NO_GT=False
            input      = None,                # predict mode: set to image path/folder
                                              # e.g. '/content/drive/MyDrive/iba/test_public_80/Color_Images'
            num_samples= 8,                   # only for visualize mode
            no_tta     = False,
            output_dir = config.OUTPUT_DIR,
        )
        return args

    # ── TERMINAL: normal argparse ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Segmentation Testing & Inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode', type=str, default='evaluate',
        choices=['evaluate', 'predict', 'visualize', 'confusion'],
        help=(
            'evaluate  : full dataset metrics\n'
            'predict   : run inference on image/folder\n'
            'visualize : side-by-side visual comparison with GT\n'
            'confusion : confusion matrix + per-class stats'
        )
    )
    parser.add_argument('--model',       type=str, default=config.MODEL_PATH,
                        help='Path to .pth checkpoint')
    parser.add_argument('--split',       type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split for evaluate/visualize/confusion')
    parser.add_argument('--input',       type=str, default=None,
                        help='Image path or folder for predict mode')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples for visualize mode')
    parser.add_argument('--no_tta',      action='store_true',
                        help='Disable Test-Time Augmentation')
    parser.add_argument('--output_dir',  type=str, default=config.OUTPUT_DIR,
                        help='Where to save outputs')
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply overrides to config
    config.MODEL_PATH  = args.model
    config.OUTPUT_DIR  = args.output_dir
    config.USE_TTA     = not args.no_tta

    print("=" * 60)
    print("SEGMENTATION TEST RUNNER")
    print("=" * 60)
    print(f"  Mode       : {args.mode}")
    print(f"  Model      : {config.MODEL_PATH}")
    print(f"  Output dir : {config.OUTPUT_DIR}")
    print(f"  TTA        : {config.USE_TTA}")
    print("=" * 60)

    # Load model
    backbone, model = load_model(config.MODEL_PATH, device)

    # Dispatch
    if args.mode == 'evaluate':
        run_evaluate(backbone, model, split=args.split)

    elif args.mode == 'predict':
        if args.input is None:
            raise ValueError("--input is required for predict mode. "
                             "Set args.input in the notebook or pass --input on CLI.")
        run_predict(backbone, model, input_path=args.input)

    elif args.mode == 'visualize':
        run_visualize(backbone, model,
                      split=args.split, num_samples=args.num_samples)

    elif args.mode == 'confusion':
        run_confusion(backbone, model, split=args.split)


# ============================================================================
# COLAB / JUPYTER USAGE
# ============================================================================
#
# Option A — just call main(), it auto-detects the notebook and reads the
# defaults you set inside parse_args() above (mode, split, input, etc.):
#
#   main()
#
# Option B — call individual functions directly:
#
#   backbone, model = load_model(config.MODEL_PATH, device)
#   run_evaluate(backbone, model, split='val')
#   run_visualize(backbone, model, split='val', num_samples=10)
#   run_confusion(backbone, model, split='val')
#   run_predict(backbone, model, input_path='/path/to/image.jpg')
#
# For predict mode in a notebook set args.input inside parse_args():
#   input = '/content/drive/MyDrive/iba/test_public_80/Color_Images'
#
# ============================================================================

if __name__ == "__main__":
    main()