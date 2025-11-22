#!/usr/bin/env python3
"""Test and enable CUDA for PyTorch"""

import subprocess
import sys
import torch
import platform

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed"""
    print("\n1. Checking NVIDIA Drivers...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ NVIDIA drivers detected")
            # Extract GPU info from nvidia-smi
            print("\n   GPU Information:")
            for line in result.stdout.split('\n')[8:13]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("   ✗ nvidia-smi failed - drivers may not be installed")
            return False
    except FileNotFoundError:
        print("   ✗ nvidia-smi not found - NVIDIA drivers not installed")
        return False
    except Exception as e:
        print(f"   ✗ Error checking drivers: {e}")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed"""
    print("\n2. Checking CUDA Toolkit...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ CUDA toolkit detected")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("   ✗ nvcc not found - CUDA toolkit not installed")
            return False
    except FileNotFoundError:
        print("   ✗ nvcc not found - CUDA toolkit not installed")
        return False
    except Exception as e:
        print(f"   ✗ Error checking CUDA toolkit: {e}")
        return False

def install_cuda_pytorch():
    """Install GPU-enabled PyTorch"""
    print("\n3. Installing GPU-enabled PyTorch...")
    print("   This may take a few minutes...")
    
    # Detect CUDA version from nvidia-smi
    cuda_version = "12.1"  # default
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                version_str = line.split(':')[1].strip()
                major_version = version_str.split('.')[0]
                if major_version == '11':
                    cuda_version = "11.8"
                elif major_version == '12':
                    cuda_version = "12.1"
                break
    except:
        pass
    
    # Map CUDA version to PyTorch index
    cuda_map = {
        "11.8": "cu118",
        "12.1": "cu121"
    }
    cu_suffix = cuda_map.get(cuda_version, "cu121")
    
    print(f"   Detected CUDA version: {cuda_version} (cu{cu_suffix})")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cu_suffix}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ PyTorch GPU installation successful")
            return True
        else:
            print("   ✗ PyTorch installation failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"   ✗ Error installing PyTorch: {e}")
        return False

def test_cuda():
    """Test CUDA functionality"""
    print("\n" + "=" * 60)
    print("CUDA Diagnostic Test")
    print("=" * 60)
    print(f"\nSystem: {platform.system()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check prerequisites
    has_drivers = check_nvidia_drivers()
    has_cuda_toolkit = check_cuda_toolkit()
    
    if not has_drivers or not has_cuda_toolkit:
        print("\n" + "=" * 60)
        print("⚠ Prerequisites missing!")
        print("=" * 60)
        print("To use GPU acceleration, you need:")
        if not has_drivers:
            print("  1. NVIDIA GPU Drivers - Download from https://www.nvidia.com/Download/driverDetails.aspx")
        if not has_cuda_toolkit:
            print("  2. NVIDIA CUDA Toolkit - Download from https://developer.nvidia.com/cuda-downloads")
        print("\nAfter installing, run this script again.")
        return
    
    # Check PyTorch CUDA support
    print("\n4. Checking PyTorch CUDA Support...")
    if "cpu" in torch.__version__:
        print("   ⚠ PyTorch is CPU-only version")
        print("   Installing GPU-enabled PyTorch...")
        if install_cuda_pytorch():
            print("\n   Please restart Python and run this script again.")
            return
        else:
            print("   ✗ Installation failed")
            return
    
    # Test CUDA availability in PyTorch
    print(f"\n5. CUDA Available in PyTorch: {torch.cuda.is_available()}")
    print(f"6. CUDA Version in PyTorch: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\n7. Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"   - Name: {torch.cuda.get_device_name(i)}")
            print(f"   - Capability: {torch.cuda.get_device_capability(i)}")
            
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            print(f"   - Memory: {props.total_memory / 1e9:.2f} GB")
        
        # Test tensor operations on GPU
        print("\n8. Testing tensor operations on GPU...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✓ Matrix multiplication on GPU successful")
        except Exception as e:
            print(f"   ✗ Error during GPU operation: {e}")
        
        # Check current device
        print(f"\n9. Current CUDA Device: {torch.cuda.current_device()}")
        print("\n" + "=" * 60)
        print("✓ CUDA is properly configured and working!")
        print("=" * 60)
        
    else:
        print("\n⚠ CUDA is still not available in PyTorch")
        print("   Please ensure all prerequisites are installed correctly")
    
    print()

if __name__ == "__main__":
    test_cuda()
