"""
================================================================================
FM MODEL LOADER - Dynamic Model Selection and Loading
================================================================================
Automatically detects and loads FM models from saved_models/FM/
Supports multiple architectures (1D U-Net, STFT 2D U-Net, etc.)
"""

import torch
from pathlib import Path
import sys

# Add src directory to path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fm.neuralnets.neuralnet import UNet1D
from fm.neuralnets.stft_unet2d import UNet2D_STFT


class FMModelLoader:
    """
    Dynamic FM model loader with architecture detection.
    """
    
    # Map of architecture names to model classes
    ARCHITECTURES = {
        '1dunet': UNet1D,
        'stft': UNet2D_STFT,
        'unet1d': UNet1D,
        'unet2d_stft': UNet2D_STFT,
    }
    
    # Default model search paths (only FinalModels - validated models)
    DEFAULT_SEARCH_PATHS = [
        'saved_models/FM/FinalModels/FM_Final_1DUNET',
        'saved_models/FM/FinalModels/FM_Final_STFT',
    ]
    
    @staticmethod
    def list_available_models(search_paths=None):
        """
        List all available FM models.
        
        Args:
            search_paths: List of directories to search (None uses defaults)
            
        Returns:
            dict: {model_path: {architecture, mode, info}}
        """
        if search_paths is None:
            search_paths = FMModelLoader.DEFAULT_SEARCH_PATHS
        
        models = {}
        
        for search_dir in search_paths:
            path = Path(search_dir)
            if not path.exists():
                continue
            
            # Find all .pth files
            for model_file in path.rglob('*.pth'):
                # Determine architecture from parent folder name
                parent_name = model_file.parent.name.lower()
                architecture = FMModelLoader._detect_architecture(parent_name)
                
                # Determine mode from filename (general, music, speech)
                mode = model_file.stem.lower()
                
                models[str(model_file)] = {
                    'architecture': architecture,
                    'mode': mode,
                    'parent': parent_name,
                    'size_mb': model_file.stat().st_size / 1024 / 1024  # MB
                }
        
        return models
    
    @staticmethod
    def _detect_architecture(folder_name):
        """Detect architecture from folder name."""
        folder_lower = folder_name.lower()
        
        if 'stft' in folder_lower or '2d' in folder_lower:
            return 'stft'
        elif '1d' in folder_lower or 'unet1d' in folder_lower:
            return '1dunet'
        else:
            # Default to 1D U-Net
            return '1dunet'
    
    @staticmethod
    def load_model(model_path=None, mode='general', architecture=None, device='cpu'):
        """
        Load FM model with automatic architecture detection.
        
        Args:
            model_path: Direct path to .pth file (None for auto-search)
            mode: Model mode ('general', 'music', 'speech')
            architecture: Force specific architecture ('1dunet', 'stft', None for auto)
            device: torch device ('cpu', 'cuda', etc.)
            
        Returns:
            tuple: (model, model_info_dict)
        """
        device_obj = torch.device(device)
        
        # If no path provided, search for model
        if model_path is None:
            model_path = FMModelLoader._find_model(mode, architecture)
            if model_path is None:
                raise FileNotFoundError(
                    f"No FM model found for mode='{mode}', architecture='{architecture}'. "
                    f"Available models: {list(FMModelLoader.list_available_models().keys())}"
                )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Detect architecture if not specified
        if architecture is None:
            parent_name = model_path.parent.name.lower()
            architecture = FMModelLoader._detect_architecture(parent_name)
        
        # Get model class
        model_class = FMModelLoader.ARCHITECTURES.get(architecture)
        if model_class is None:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Supported: {list(FMModelLoader.ARCHITECTURES.keys())}"
            )
        
        # Instantiate model
        model = model_class(in_channels=1, out_channels=1).to(device_obj)
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=device_obj)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    info = {
                        'val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A')),
                        'epoch': checkpoint.get('epoch', 'N/A'),
                    }
                else:
                    # Assume checkpoint is the state dict itself
                    model.load_state_dict(checkpoint)
                    info = {}
            else:
                # Old format - direct state dict
                model.load_state_dict(checkpoint)
                info = {}
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")
        
        model.eval()
        
        # Build info dict
        model_info = {
            'path': str(model_path),
            'architecture': architecture,
            'mode': model_path.stem,
            'size_mb': model_path.stat().st_size / 1024 / 1024,
            **info
        }
        
        return model, model_info
    
    @staticmethod
    def _find_model(mode, architecture):
        """Find model matching mode and architecture."""
        models = FMModelLoader.list_available_models()
        
        # Filter by mode
        mode_matches = [
            path for path, info in models.items()
            if info['mode'] == mode.lower()
        ]
        
        if not mode_matches:
            # No exact mode match, try any model
            mode_matches = list(models.keys())
        
        # Filter by architecture if specified
        if architecture:
            arch_matches = [
                path for path in mode_matches
                if models[path]['architecture'] == architecture
            ]
            if arch_matches:
                return arch_matches[0]
        
        # Return first match
        return mode_matches[0] if mode_matches else None
    
    @staticmethod
    def print_available_models():
        """Print all available models in a nice format."""
        models = FMModelLoader.list_available_models()
        
        if not models:
            print("❌ No FM models found!")
            print(f"   Search paths: {FMModelLoader.DEFAULT_SEARCH_PATHS}")
            return
        
        print("\n" + "="*80)
        print("AVAILABLE FM MODELS")
        print("="*80)
        
        # Group by architecture
        by_arch = {}
        for path, info in models.items():
            arch = info['architecture']
            if arch not in by_arch:
                by_arch[arch] = []
            by_arch[arch].append((path, info))
        
        for arch, model_list in by_arch.items():
            print(f"\n📦 {arch.upper()} Architecture:")
            for path, info in model_list:
                print(f"   • {info['mode']:10s} | {info['size_mb']:6.2f} MB | {path}")
        
        print("="*80 + "\n")


def get_model_for_inference(mode='general', architecture=None, device='auto'):
    """
    Convenience function for getting a model ready for inference.
    
    Args:
        mode: 'general', 'music', or 'speech'
        architecture: '1dunet', 'stft', or None (auto-detect)
        device: 'auto', 'cpu', 'cuda', or torch.device
        
    Returns:
        tuple: (model, model_info)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return FMModelLoader.load_model(mode=mode, architecture=architecture, device=device)


if __name__ == '__main__':
    # Test the model loader
    print("Testing FM Model Loader...")
    FMModelLoader.print_available_models()
    
    # Try loading each architecture
    for arch in ['1dunet', 'stft']:
        try:
            print(f"\nLoading {arch} model (general mode)...")
            model, info = get_model_for_inference(mode='general', architecture=arch, device='cpu')
            print(f"✅ Loaded: {info['architecture']} - {info['mode']}")
            print(f"   Path: {info['path']}")
            print(f"   Size: {info['size_mb']:.2f} MB")
            if 'val_loss' in info:
                print(f"   Val Loss: {info['val_loss']}")
        except Exception as e:
            print(f"⚠️  Could not load {arch}: {e}")
