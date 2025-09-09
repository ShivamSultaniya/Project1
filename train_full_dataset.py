#!/usr/bin/env python3
"""
Full Dataset Multi-Input Image Segmentation Training Script
This script trains on the complete dataset (2968 train + 1694 test images) with advanced features:
- Data augmentation for better generalization
- Model checkpointing and resuming
- Advanced validation metrics
- Learning rate scheduling
- Early stopping
- Mixed precision training for faster GPU training
- Comprehensive logging and visualization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import random
from tqdm import tqdm
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdvancedDualInputDataset(Dataset):
    """Enhanced dataset class with data augmentation"""
    
    def __init__(self, data_dir, image_files, transform=None, augment_transform=None, is_train=True):
        self.data_dir = data_dir
        self.image_files = image_files
        self.transform = transform
        self.augment_transform = augment_transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load dual input images
        im1_path = os.path.join(self.data_dir, 'im1', img_name)
        im2_path = os.path.join(self.data_dir, 'im2', img_name)
        
        im1 = Image.open(im1_path).convert('RGB')
        im2 = Image.open(im2_path).convert('RGB')
        
        # Load dual labels
        label1_path = os.path.join(self.data_dir, 'label1', img_name)
        label2_path = os.path.join(self.data_dir, 'label2', img_name)
        
        label1 = Image.open(label1_path).convert('L')
        label2 = Image.open(label2_path).convert('L')
        
        # Apply augmentations if training
        if self.is_train and self.augment_transform and random.random() > 0.5:
            # Apply same augmentation to both images and labels
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.augment_transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.augment_transform(im2)
            
            random.seed(seed)
            torch.manual_seed(seed)
            label1 = self.augment_transform(label1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            label2 = self.augment_transform(label2)
        
        # Apply standard transforms
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            
            label_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            label1 = label_transform(label1)
            label2 = label_transform(label2)
        
        # Normalize labels to 0, 1, 2 classes
        label1 = (label1 * 255).long()
        label2 = (label2 * 255).long()
        
        # Create proper class mapping: 0->0, 128->1, 255->2
        label1_new = torch.zeros_like(label1)
        label2_new = torch.zeros_like(label2)
        
        label1_new[label1 == 128] = 1
        label1_new[label1 == 255] = 2
        label2_new[label2 == 128] = 1
        label2_new[label2 == 255] = 2
        
        label1 = label1_new
        label2 = label2_new
        
        return {
            'im1': im1,
            'im2': im2,
            'label1': label1.squeeze(0),
            'label2': label2.squeeze(0),
            'filename': img_name
        }

class EnhancedDualInputUNet(nn.Module):
    """Enhanced U-Net with attention mechanisms and skip connections"""
    
    def __init__(self, in_channels=6, out_channels=3, features=[64, 128, 256, 512]):
        super(EnhancedDualInputUNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder 1
        for feature in reversed(features):
            self.decoder1.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder1.append(self._conv_block(feature * 2, feature))
        
        # Decoder 2
        for feature in reversed(features):
            self.decoder2.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder2.append(self._conv_block(feature * 2, feature))
        
        # Final layers
        self.final_conv1 = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        # Concatenate dual inputs
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        skip_connections = skip_connections[::-1]
        
        # Decoder 1
        x1_dec = x
        for idx in range(0, len(self.decoder1), 2):
            x1_dec = self.decoder1[idx](x1_dec)
            skip_connection = skip_connections[idx // 2]
            
            if x1_dec.shape != skip_connection.shape:
                x1_dec = transforms.functional.resize(x1_dec, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x1_dec), dim=1)
            x1_dec = self.decoder1[idx + 1](concat_skip)
        
        # Decoder 2
        x2_dec = x
        for idx in range(0, len(self.decoder2), 2):
            x2_dec = self.decoder2[idx](x2_dec)
            skip_connection = skip_connections[idx // 2]
            
            if x2_dec.shape != skip_connection.shape:
                x2_dec = transforms.functional.resize(x2_dec, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x2_dec), dim=1)
            x2_dec = self.decoder2[idx + 1](concat_skip)
        
        return self.final_conv1(x1_dec), self.final_conv2(x2_dec)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_metrics(pred, target, num_classes=3):
    """Calculate comprehensive metrics"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # IoU per class
    ious = []
    f1s = []
    
    for cls in range(num_classes):
        pred_cls = (pred_np == cls)
        target_cls = (target_np == cls)
        
        if target_cls.sum() == 0:  # No ground truth for this class
            if pred_cls.sum() == 0:
                iou = 1.0
                f1 = 1.0
            else:
                iou = 0.0
                f1 = 0.0
        else:
            intersection = np.logical_and(pred_cls, target_cls).sum()
            union = np.logical_or(pred_cls, target_cls).sum()
            iou = intersection / union if union > 0 else 0.0
            
            # F1 score
            precision = intersection / pred_cls.sum() if pred_cls.sum() > 0 else 0.0
            recall = intersection / target_cls.sum() if target_cls.sum() > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        ious.append(iou)
        f1s.append(f1)
    
    return {
        'mean_iou': np.mean(ious),
        'mean_f1': np.mean(f1s),
        'class_ious': ious,
        'class_f1s': f1s
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_iou'], checkpoint['loss']

def get_system_info():
    """Get system information for optimal configuration"""
    info = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['gpu_available']:
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Optimal batch size based on GPU memory
        if info['gpu_memory'] >= 16:
            info['batch_size'] = 16
        elif info['gpu_memory'] >= 8:
            info['batch_size'] = 8
        elif info['gpu_memory'] >= 4:
            info['batch_size'] = 4
        else:
            info['batch_size'] = 2
    else:
        info['batch_size'] = 2
    
    # Optimal number of workers
    import multiprocessing
    info['num_workers'] = min(8, multiprocessing.cpu_count())
    
    return info

def train_full_dataset():
    """Main training function for full dataset"""
    
    print("🚀 Full Dataset Training Script")
    print("=" * 50)
    
    # System configuration
    system_info = get_system_info()
    device = torch.device('cuda' if system_info['gpu_available'] else 'cpu')
    
    print(f"Device: {device}")
    if system_info['gpu_available']:
        print(f"GPU: {system_info['gpu_name']}")
        print(f"GPU Memory: {system_info['gpu_memory']:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    batch_size = system_info['batch_size']
    num_workers = system_info['num_workers']
    
    print(f"Batch Size: {batch_size}")
    print(f"Workers: {num_workers}")
    print("=" * 50)
    
    # Data transforms
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Augmentation transforms for training
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    # Load all image files
    train_files = sorted([f for f in os.listdir('train/im1') if f.endswith('.png')])
    test_files = sorted([f for f in os.listdir('test/im1') if f.endswith('.png')])
    
    print(f"📊 Dataset Statistics:")
    print(f"Training images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")
    print(f"Total images: {len(train_files) + len(test_files)}")
    print("=" * 50)
    
    # Create datasets
    train_dataset = AdvancedDualInputDataset(
        'train', train_files, 
        transform=base_transform, 
        augment_transform=augment_transform,
        is_train=True
    )
    test_dataset = AdvancedDualInputDataset(
        'test', test_files, 
        transform=base_transform, 
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Initialize model
    model = EnhancedDualInputUNet(in_channels=6, out_channels=3).to(device)
    
    # Model compilation for PyTorch 2.0+
    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)
    
    # Loss function and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)  # Better for imbalanced classes
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training configuration
    num_epochs = 50
    best_iou = 0.0
    patience = 10
    patience_counter = 0
    
    # Logging
    train_losses = []
    val_losses = []
    val_ious = []
    val_f1s = []
    
    print("🏋️ Starting Full Dataset Training...")
    print(f"Epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print("=" * 50)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            label1 = batch['label1'].to(device, non_blocking=True)
            label2 = batch['label2'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:  # Mixed precision
                with autocast():
                    out1, out2 = model(im1, im2)
                    loss1 = criterion(out1, label1)
                    loss2 = criterion(out2, label2)
                    total_loss = loss1 + loss2
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Regular precision
                out1, out2 = model(im1, im2)
                loss1 = criterion(out1, label1)
                loss2 = criterion(out2, label2)
                total_loss = loss1 + loss2
                
                total_loss.backward()
                optimizer.step()
            
            running_loss += total_loss.item()
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics = {'mean_iou': [], 'mean_f1': []}
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                im1 = batch['im1'].to(device, non_blocking=True)
                im2 = batch['im2'].to(device, non_blocking=True)
                label1 = batch['label1'].to(device, non_blocking=True)
                label2 = batch['label2'].to(device, non_blocking=True)
                
                if scaler:
                    with autocast():
                        out1, out2 = model(im1, im2)
                        loss1 = criterion(out1, label1)
                        loss2 = criterion(out2, label2)
                        total_loss = loss1 + loss2
                else:
                    out1, out2 = model(im1, im2)
                    loss1 = criterion(out1, label1)
                    loss2 = criterion(out2, label2)
                    total_loss = loss1 + loss2
                
                val_loss += total_loss.item()
                
                # Calculate metrics
                pred1 = torch.argmax(out1, dim=1)
                pred2 = torch.argmax(out2, dim=1)
                
                metrics1 = calculate_metrics(pred1, label1)
                metrics2 = calculate_metrics(pred2, label2)
                
                avg_iou = (metrics1['mean_iou'] + metrics2['mean_iou']) / 2
                avg_f1 = (metrics1['mean_f1'] + metrics2['mean_f1']) / 2
                
                all_metrics['mean_iou'].append(avg_iou)
                all_metrics['mean_f1'].append(avg_f1)
                
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'IoU': f'{avg_iou:.4f}'
                })
        
        # Calculate epoch metrics
        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_iou = np.mean(all_metrics['mean_iou'])
        epoch_val_f1 = np.mean(all_metrics['mean_f1'])
        
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)
        val_f1s.append(epoch_val_f1)
        
        # Print epoch results
        elapsed_time = time.time() - start_time
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {elapsed_time/60:.1f}min')
        print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        print(f'Val IoU: {epoch_val_iou:.4f} | Val F1: {epoch_val_f1:.4f}')
        
        # Save best model
        if epoch_val_iou > best_iou:
            best_iou = epoch_val_iou
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_iou, epoch_val_loss,
                'best_full_dataset_model.pth'
            )
            print(f'🎉 New best model saved! IoU: {best_iou:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'⏹️  Early stopping triggered after {patience} epochs without improvement')
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_iou, epoch_val_loss,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        print("-" * 50)
    
    # Training completed
    total_time = time.time() - start_time
    print(f'\n🎉 Training completed in {total_time/3600:.2f} hours')
    print(f'Best validation IoU: {best_iou:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), 'final_full_dataset_model.pth')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_f1s': val_f1s,
        'best_iou': best_iou,
        'total_epochs': len(train_losses),
        'total_time_hours': total_time / 3600,
        'system_info': system_info
    }
    
    with open('training_history_full_dataset.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Validation IoU', color='green')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Validation F1', color='orange')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('full_dataset_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\n📊 Training artifacts saved:')
    print(f'- Best model: best_full_dataset_model.pth')
    print(f'- Final model: final_full_dataset_model.pth')
    print(f'- Training history: training_history_full_dataset.json')
    print(f'- Training curves: full_dataset_training_curves.png')

if __name__ == "__main__":
    # Verify dataset structure
    required_dirs = ['train/im1', 'train/im2', 'train/label1', 'train/label2',
                     'test/im1', 'test/im2', 'test/label1', 'test/label2']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory '{dir_path}' not found!")
            exit(1)
    
    print("✅ Dataset structure verified.")
    train_full_dataset() 