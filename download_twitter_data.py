"""
Script to download Twitter datasets from Zenodo
WARNING: These files are very large (3+ GB total) and will take time to download.
"""
import urllib.request
from pathlib import Path
import os
import sys

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# URLs from README.md
TWITTER_DATASETS = {
    "depression.zip": "https://zenodo.org/records/5854911/files/depression.zip?download=1",
    "neg.zip": "https://zenodo.org/records/5854911/files/neg.zip?download=1"
}

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_file_size(url):
    """Get file size from URL without downloading"""
    try:
        req = urllib.request.Request(url)
        req.add_header('Range', 'bytes=0-0')
        with urllib.request.urlopen(req) as response:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except:
        pass
    return None

def download_file(url, filename):
    """Download a file with progress bar"""
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ {filename} already exists ({format_size(filepath.stat().st_size)})")
        print(f"   Skipping download (file already exists)")
        return True
    
    # Try to get file size
    file_size = get_file_size(url)
    if file_size:
        print(f"📦 File size: {format_size(file_size)}")
    
    print(f"📥 Downloading {filename}...")
    print(f"   URL: {url}")
    print(f"   ⚠️  This may take a while (large file)...")
    
    try:
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({format_size(downloaded)}/{format_size(total_size)})", end="")
            else:
                mb_downloaded = block_num * block_size / (1024 * 1024)
                print(f"\r   Progress: {format_size(block_num * block_size)} downloaded...", end="")
        
        urllib.request.urlretrieve(url, filepath, show_progress)
        print(f"\n✅ Downloaded {filename}")
        if filepath.exists():
            print(f"   Saved: {format_size(filepath.stat().st_size)}")
        return True
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Download interrupted by user")
        if filepath.exists():
            filepath.unlink()
        return False
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("DOWNLOADING TWITTER DATASETS")
    print("=" * 70)
    print(f"\n⚠️  WARNING: These files are VERY LARGE (3+ GB total)")
    print(f"   Download time will depend on your internet connection")
    print(f"   You can cancel anytime with Ctrl+C\n")
    print("Proceeding with download...\n")
    
    print(f"📁 Target directory: {DATA_DIR.absolute()}\n")
    
    success_count = 0
    for filename, url in TWITTER_DATASETS.items():
        print(f"\n{'='*70}")
        if download_file(url, filename):
            success_count += 1
        else:
            print(f"❌ Failed to download {filename}")
        print()
    
    print("=" * 70)
    if success_count == len(TWITTER_DATASETS):
        print("✅ All datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Extract the zip files:")
        print("   - unzip data/depression.zip -d data/depression")
        print("   - unzip data/neg.zip -d data/neg")
        print("\n2. Then run the preprocessing pipeline:")
        print("   - python src/process_posclass.py")
        print("   - python src/process_negclass.py")
        print("   - python src/final_df.py")
        print("   - python src/short_df.py")
        print("   - python src/preprocessing.py")
        print("   - python src/features_shorter2.py")
    else:
        print(f"⚠️  Downloaded {success_count}/{len(TWITTER_DATASETS)} files")
        if success_count > 0:
            print("   You can re-run this script to download the remaining files")
    print("=" * 70)


