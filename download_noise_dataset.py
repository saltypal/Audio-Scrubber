"""
Download Real Noise Dataset for Training

This script downloads the DEMAND database - a collection of real-world environmental noise.
Run this BEFORE training to ensure the model learns on realistic noise.

Usage:
    python download_noise_dataset.py
"""

import urllib.request
import zipfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import Paths

def download_demand_noise():
    """Download DEMAND noise dataset"""
    
    print("\n" + "="*80)
    print("Downloading DEMAND Noise Dataset")
    print("="*80 + "\n")
    
    # Create noise directory
    Paths.NOISE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # DEMAND dataset URL (subset - cafeteria noise)
    # Full dataset: https://zenodo.org/record/1227121
    url = "https://zenodo.org/record/1227121/files/PCAFETER.zip?download=1"
    
    zip_path = Paths.NOISE_ROOT / "demand_noise.zip"
    
    print(f"📥 Downloading from Zenodo...")
    print(f"   URL: {url}")
    print(f"   This may take a few minutes...\n")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✅ Download complete!")
        
        print(f"\n📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Paths.NOISE_ROOT)
        
        print(f"✅ Extraction complete!")
        
        # Clean up zip
        zip_path.unlink()
        
        # Count noise files
        noise_files = list(Paths.NOISE_ROOT.rglob('*.wav')) + list(Paths.NOISE_ROOT.rglob('*.flac'))
        print(f"\n✅ Downloaded {len(noise_files)} noise files")
        print(f"   Location: {Paths.NOISE_ROOT}")
        
        print("\n" + "="*80)
        print("✅ Noise dataset ready!")
        print("="*80)
        print("\nYou can now train your model with realistic noise.")
        print("Run: python src/model/backshot_onfly.py --epochs 10")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n📝 Manual Download Instructions:")
        print("1. Visit: https://zenodo.org/record/1227121")
        print("2. Download one or more noise files")
        print(f"3. Place .wav files in: {Paths.NOISE_ROOT}")
        print("\nOr use YouTube:")
        print("- Search for 'room noise' or 'ambient noise'")
        print("- Download audio")
        print(f"- Convert to .flac and place in: {Paths.NOISE_ROOT}")


if __name__ == "__main__":
    download_demand_noise()
