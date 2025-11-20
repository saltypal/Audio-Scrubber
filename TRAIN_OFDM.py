#!/usr/bin/env python
"""
Quick Start: Train OFDM Denoising Model

This script provides an easy entry point for training.
Just run: python TRAIN_OFDM.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         OFDM DENOISING MODEL - TRAINING LAUNCHER         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if dataset exists
    dataset_path = Path("dataset/OFDM/clean_ofdm.iq")
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        print("\n💡 Generate dataset first:")
        print("   python dataset_ofdm/ofdm_dataset_creation.py")
        return
    
    print("✅ Dataset found!")
    print(f"   Size: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # Training options
    print("🎯 Training Options:")
    print("   [1] Quick Test (10 epochs, batch 16)")
    print("   [2] Standard Training (50 epochs, batch 32)")
    print("   [3] Full Training (100 epochs, batch 32)")
    print("   [4] Custom (specify parameters)")
    print("   [5] Exit")
    
    choice = input("\n👉 Select option [1-5]: ").strip()
    
    if choice == "1":
        cmd = ["python", "src/ofdm/model/train_ofdm.py", "--epochs", "10", "--batch_size", "16"]
    elif choice == "2":
        cmd = ["python", "src/ofdm/model/train_ofdm.py", "--epochs", "50", "--batch_size", "32"]
    elif choice == "3":
        cmd = ["python", "src/ofdm/model/train_ofdm.py", "--epochs", "100", "--batch_size", "32"]
    elif choice == "4":
        epochs = input("   Epochs [50]: ").strip() or "50"
        batch = input("   Batch size [32]: ").strip() or "32"
        lr = input("   Learning rate [0.001]: ").strip() or "0.001"
        cmd = ["python", "src/ofdm/model/train_ofdm.py", 
               "--epochs", epochs, "--batch_size", batch, "--lr", lr]
    elif choice == "5":
        print("\n👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice!")
        return
    
    print("\n🚀 Starting training...\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
