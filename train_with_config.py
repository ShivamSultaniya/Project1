#!/usr/bin/env python3
"""
Config-Driven Multi-Input Image Segmentation Training Script
This script loads all parameters from config.yaml for maximum flexibility.
"""

import os
import sys
import yaml
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
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ConfigurableDataset(Dataset):
    """Dataset class that uses configuration parameters"""
    
    def __init__(self, data_dir, image_files, config, is_train=True):
        self.data_dir = data_dir
        self.image_files = image_files
        self.config = config
        self.is_train = is_train
        
        # Build transforms from config
        self.transform = self._build_transform()
        self.augment_transform = self._build_augmentation() if is_train else None
        
    def _build_transform(self):
        """Build base transform from config"""
        transforms_list = [
            transforms.Resize(tuple(self.config['dataset']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transforms_list)
    
    def _build_augmentation(self):
        """Build augmentation transforms from config"""
        if not self.config['augmentation']['enabled']:
            return None
            
        aug_config = self.config['augmentation']['transforms']
        transforms_list = []
        
        if aug_config['horizontal_flip']['enabled']:
            transforms_list.append(
                transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']['probability'])
            )
        
        if aug_config['vertical_flip']['enabled']:
            transforms_list.append(
                transforms.RandomVerticalFlip(p=aug_config['vertical_flip']['probability'])
            )
        
        if aug_config['rotation']['enabled']:
            transforms_list.append(
                transforms.RandomRotation(degrees=aug_config['rotation']['degrees'])
            )
        
        if aug_config['color_jitter']['enabled']:
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=aug_config['color_jitter']['brightness'],
                    contrast=aug_config['color_jitter']['contrast'],
                    saturation=aug_config['color_jitter']['saturation'],
                    hue=aug_config['color_jitter']['hue']
                )
            )
        
        return transforms.Compose(transforms_list) if transforms_list else None
        
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
        if self.is_train and self.augment_transform and random.random() < self.config['augmentation']['probability']:
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
        im1 = self.transform(im1)
        im2 = self.transform(im2)
        
        label_transform = transforms.Compose([
            transforms.Resize(tuple(self.config['dataset']['image_size'])),
            transforms.ToTensor()
        ])
        label1 = label_transform(label1)
        label2 = label_transform(label2)
        
        # Normalize labels to class indices
        label1 = (label1 * 255).long()
        label2 = (label2 * 255).long()
        
        # Create proper class mapping from config
        class_mapping = self.config['dataset']['class_mapping']
        label1_new = torch.zeros_like(label1)
        label2_new = torch.zeros_like(label2)
        
        label1_new[label1 == 128] = class_mapping['class_1']
        label1_new[label1 == 255] = class_mapping['class_2']
        label2_new[label2 == 128] = class_mapping['class_1']
        label2_new[label2 == 255] = class_mapping['class_2']
        
        return {
            'im1': im1,
            'im2': im2,
            'label1': label1_new.squeeze(0),
            'label2': label2_new.squeeze(0),
            'filename': img_name
        }

class ConfigurableUNet(nn.Module):
    """U-Net model configured from config file"""
    
    def __init__(self, config):
        super(ConfigurableUNet, self).__init__()
        
        model_config = config['model']
        self.features = model_config['features']
        in_channels = model_config['input_channels']
        out_channels = model_config['output_channels']
        dropout_rate = model_config['dropout_rate']
        
        self.encoder = nn.ModuleList()
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in self.features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(self.features[-1], self.features[-1] * 2)
        
        # Decoder 1
        for feature in reversed(self.features):
            self.decoder1.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder1.append(self._conv_block(feature * 2, feature))
        
        # Decoder 2
        for feature in reversed(self.features):
            self.decoder2.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder2.append(self._conv_block(feature * 2, feature))
        
        # Final layers
        self.final_conv1 = nn.Conv2d(self.features[0], out_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(self.features[0], out_channels, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
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
    """Focal Loss implementation"""
    
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

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Config file '{config_path}' not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

def setup_logging(config):
    """Setup logging based on config"""
    log_config = config['logging']
    
    if not log_config['enabled']:
        return
    
    # Configure logging
    log_level = getattr(logging, log_config['log_level'])
    
    handlers = []
    
    if log_config['console_logging']:
        handlers.append(logging.StreamHandler())
    
    if log_config['log_to_file']:
        os.makedirs(os.path.dirname(log_config['log_file']), exist_ok=True)
        handlers.append(logging.FileHandler(log_config['log_file']))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def get_device_and_batch_size(config):
    """Get device and batch size from config"""
    hardware_config = config['hardware']
    
    # Device selection
    if hardware_config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(hardware_config['device'])
    
    # Batch size selection
    batch_size = config['training']['batch_size']
    if batch_size == 'auto':
        if device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 16:
                batch_size = 16
            elif gpu_memory >= 8:
                batch_size = 8
            elif gpu_memory >= 4:
                batch_size = 4
            else:
                batch_size = 2
        else:
            batch_size = 2
    
    return device, batch_size

def create_optimizer(model, config):
    """Create optimizer from config"""
    train_config = config['training']
    optimizer_name = train_config['optimizer']
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer

def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    train_config = config['training']
    scheduler_name = train_config['scheduler']
    scheduler_params = train_config['scheduler_params']
    
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            T_mult=scheduler_params['T_mult'],
            eta_min=scheduler_params['eta_min']
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            eta_min=scheduler_params['eta_min']
        )
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler

def create_loss_function(config):
    """Create loss function from config"""
    train_config = config['training']
    loss_name = train_config['loss_function']
    
    if loss_name == 'FocalLoss':
        focal_params = train_config['focal_loss_params']
        criterion = FocalLoss(
            alpha=focal_params['alpha'],
            gamma=focal_params['gamma']
        )
    elif loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    return criterion

def calculate_metrics(pred, target, num_classes=3):
    """Calculate comprehensive metrics"""
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    ious = []
    f1s = []
    
    for cls in range(num_classes):
        pred_cls = (pred_np == cls)
        target_cls = (target_np == cls)
        
        if target_cls.sum() == 0:
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

def train_with_config(config_path='config.yaml'):
    """Main training function using configuration"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logging.info("🚀 Starting Config-Driven Training")
    
    # Set random seeds
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if config['experiment']['deterministic']:
        torch.backends.cudnn.deterministic = True
    if config['experiment']['benchmark']:
        torch.backends.cudnn.benchmark = True
    
    # Get device and batch size
    device, batch_size = get_device_and_batch_size(config)
    
    print(f"📊 Experiment: {config['experiment']['name']} v{config['experiment']['version']}")
    print(f"📝 Description: {config['experiment']['description']}")
    print(f"🔧 Device: {device}")
    print(f"📦 Batch Size: {batch_size}")
    
    # Load dataset
    dataset_config = config['dataset']
    train_files = sorted([f for f in os.listdir(f"{dataset_config['train_dir']}/im1") if f.endswith('.png')])
    test_files = sorted([f for f in os.listdir(f"{dataset_config['test_dir']}/im1") if f.endswith('.png')])
    
    # Apply debug limits if enabled
    if config['debug']['enabled']:
        if config['debug']['limit_train_samples']:
            train_files = train_files[:config['debug']['limit_train_samples']]
        if config['debug']['limit_test_samples']:
            test_files = test_files[:config['debug']['limit_test_samples']]
    
    print(f"📊 Training images: {len(train_files)}")
    print(f"📊 Test images: {len(test_files)}")
    
    # Create datasets
    train_dataset = ConfigurableDataset(dataset_config['train_dir'], train_files, config, is_train=True)
    test_dataset = ConfigurableDataset(dataset_config['test_dir'], test_files, config, is_train=False)
    
    # Create data loaders
    hardware_config = config['hardware']
    num_workers = hardware_config['num_workers']
    if num_workers == 'auto':
        import multiprocessing
        num_workers = min(8, multiprocessing.cpu_count())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=hardware_config['pin_memory'] and device.type == 'cuda',
        persistent_workers=hardware_config['persistent_workers'] and num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=hardware_config['pin_memory'] and device.type == 'cuda',
        persistent_workers=hardware_config['persistent_workers'] and num_workers > 0
    )
    
    # Create model
    model = ConfigurableUNet(config).to(device)
    
    # Model compilation
    if hardware_config['compile_model'] and hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)
    
    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_loss_function(config)
    
    # Mixed precision training
    scaler = GradScaler() if hardware_config['mixed_precision'] and device.type == 'cuda' else None
    
    # Training configuration
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    
    # Early stopping
    early_stopping_config = train_config['early_stopping']
    best_metric = 0.0 if early_stopping_config['mode'] == 'max' else float('inf')
    patience_counter = 0
    
    # Logging
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print("🏋️ Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            label1 = batch['label1'].to(device, non_blocking=True)
            label2 = batch['label2'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    out1, out2 = model(im1, im2)
                    loss1 = criterion(out1, label1)
                    loss2 = criterion(out2, label2)
                    total_loss = loss1 + loss2
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out1, out2 = model(im1, im2)
                loss1 = criterion(out1, label1)
                loss2 = criterion(out2, label2)
                total_loss = loss1 + loss2
                
                total_loss.backward()
                optimizer.step()
            
            running_loss += total_loss.item()
            train_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()
        
        # Validation phase
        if config['validation']['enabled'] and (epoch + 1) % config['validation']['frequency'] == 0:
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
            
            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_iou = np.mean(all_metrics['mean_iou'])
            epoch_val_f1 = np.mean(all_metrics['mean_f1'])
            
            val_losses.append(epoch_val_loss)
            val_metrics.append({'iou': epoch_val_iou, 'f1': epoch_val_f1})
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
            print(f'Val IoU: {epoch_val_iou:.4f} | Val F1: {epoch_val_f1:.4f}')
            
            # Early stopping check
            if early_stopping_config['enabled']:
                current_metric = epoch_val_iou if early_stopping_config['monitor'] == 'val_iou' else epoch_val_loss
                
                if early_stopping_config['mode'] == 'max':
                    if current_metric > best_metric + early_stopping_config['min_delta']:
                        best_metric = current_metric
                        patience_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), 'best_config_model.pth')
                        print(f'🎉 New best model saved! {early_stopping_config["monitor"]}: {best_metric:.4f}')
                    else:
                        patience_counter += 1
                else:
                    if current_metric < best_metric - early_stopping_config['min_delta']:
                        best_metric = current_metric
                        patience_counter = 0
                        torch.save(model.state_dict(), 'best_config_model.pth')
                        print(f'🎉 New best model saved! {early_stopping_config["monitor"]}: {best_metric:.4f}')
                    else:
                        patience_counter += 1
                
                if patience_counter >= early_stopping_config['patience']:
                    print(f'⏹️ Early stopping triggered after {early_stopping_config["patience"]} epochs')
                    break
    
    # Save final model
    torch.save(model.state_dict(), 'final_config_model.pth')
    
    # Save training history
    if config['logging']['save_training_history']:
        history = {
            'config': config,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'best_metric': best_metric,
            'total_epochs': len(train_losses),
            'training_time_hours': (time.time() - start_time) / 3600
        }
        
        with open(config['logging']['history_file'], 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    # Plot training curves if enabled
    if config['logging']['plot_training_curves'] and val_losses:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if val_metrics:
            plt.subplot(1, 3, 2)
            val_ious = [m['iou'] for m in val_metrics]
            plt.plot(val_ious, label='Validation IoU', color='green')
            plt.title('Validation IoU')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            val_f1s = [m['f1'] for m in val_metrics]
            plt.plot(val_f1s, label='Validation F1', color='orange')
            plt.title('Validation F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('config_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    total_time = time.time() - start_time
    print(f'\n🎉 Training completed in {total_time/3600:.2f} hours')
    print(f'📊 Best {early_stopping_config["monitor"]}: {best_metric:.4f}')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Config-driven training script')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Verify dataset structure
    required_dirs = ['train/im1', 'train/im2', 'train/label1', 'train/label2',
                     'test/im1', 'test/im2', 'test/label1', 'test/label2']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory '{dir_path}' not found!")
            sys.exit(1)
    
    print("✅ Dataset structure verified.")
    train_with_config(args.config) 