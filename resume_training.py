#!/usr/bin/env python3
"""
Resume Training Script
This script allows you to resume training from a saved checkpoint.
"""

import os
import torch
from train_full_dataset import EnhancedDualInputUNet, train_full_dataset, load_checkpoint

def resume_from_checkpoint(checkpoint_path):
    """Resume training from a checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file '{checkpoint_path}' not found!")
        return
    
    print(f"🔄 Resuming training from: {checkpoint_path}")
    
    # Load checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Checkpoint Info:")
    print(f"- Epoch: {checkpoint['epoch'] + 1}")
    print(f"- Best IoU: {checkpoint['best_iou']:.4f}")
    print(f"- Loss: {checkpoint['loss']:.4f}")
    print(f"- Timestamp: {checkpoint['timestamp']}")
    
    # You can modify the train_full_dataset function to accept resume parameters
    # For now, this shows how to load the checkpoint
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedDualInputUNet(in_channels=6, out_channels=3).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model state loaded successfully!")
    
    print("\n🔧 To fully resume training, you would need to:")
    print("1. Modify train_full_dataset() to accept resume parameters")
    print("2. Load optimizer and scheduler states")
    print("3. Start from the saved epoch")
    print("4. Use the saved best_iou for comparison")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python resume_training.py <checkpoint_path>")
        print("Example: python resume_training.py checkpoint_epoch_10.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    resume_from_checkpoint(checkpoint_path) 