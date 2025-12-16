"""
Script to download Reddit datasets for the intermediate presentation
These are the datasets mentioned in the notebook.
"""
import urllib.request
from pathlib import Path
import os

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# URLs from the README and notebook
REDDIT_DATASETS = {
    "depression_pre_features_tfidf_256.csv": "https://zenodo.org/records/3941387/files/depression_pre_features_tfidf_256.csv?download=1",
    "fitness_pre_features_tfidf_256.csv": "https://zenodo.org/records/3941387/files/fitness_pre_features_tfidf_256.csv?download=1"
}

def download_file(url, filename):
    """Download a file with progress bar"""
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ {filename} already exists ({size_mb:.2f} MB)")
        return True
    
    print(f"📥 Downloading {filename}...")
    print(f"   URL: {url}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, filepath, show_progress)
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING REDDIT DATASETS")
    print("=" * 60)
    print(f"\nTarget directory: {DATA_DIR.absolute()}\n")
    
    success_count = 0
    for filename, url in REDDIT_DATASETS.items():
        if download_file(url, filename):
            success_count += 1
        print()
    
    print("=" * 60)
    if success_count == len(REDDIT_DATASETS):
        print("✅ All datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Run: python intermediate_presentation/create_parquets.py")
        print("   This will create processed parquet files in data/")
    else:
        print(f"⚠️  Downloaded {success_count}/{len(REDDIT_DATASETS)} files")
    print("=" * 60)


