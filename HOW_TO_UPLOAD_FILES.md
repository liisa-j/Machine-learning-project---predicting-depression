# How to Upload/Copy Files from Your PC to This Project

## Method 1: Using Windows File Explorer (Easiest)

### Step 1: Open File Explorer
- Press `Windows Key + E` or click the File Explorer icon

### Step 2: Navigate to Your Data Files
- Find where your data files are stored on your computer
- This could be in Downloads, Desktop, Documents, or another folder

### Step 3: Copy the Files
1. Select the files/folders you want to copy
2. Right-click and choose **"Copy"** (or press `Ctrl + C`)

### Step 4: Navigate to Project Data Folder
Open this path in File Explorer:
```
C:\Users\showkat\mlCLASS\Machine-learning-project---predicting-depression\data
```

**Quick way to get there:**
- Copy this path: `C:\Users\showkat\mlCLASS\Machine-learning-project---predicting-depression\data`
- Paste it into the address bar at the top of File Explorer
- Press Enter

### Step 5: Paste the Files
- Right-click in the `data` folder and choose **"Paste"** (or press `Ctrl + V`)

## Method 2: Drag and Drop

1. Open two File Explorer windows:
   - Window 1: Your data files location
   - Window 2: The project `data` folder (`C:\Users\showkat\mlCLASS\Machine-learning-project---predicting-depression\data`)

2. Drag files from Window 1 to Window 2

## Method 3: Using Command Line (PowerShell)

If you know the path to your files, I can help you copy them using commands.

## What Files Should You Copy?

### If you have ZIP files:
- `depression.zip` → Copy to `data/` folder
- `neg.zip` → Copy to `data/` folder
- Then I'll help you extract them

### If you have already extracted folders:
- Copy the entire `depression` folder to `data/depression/`
- Copy the entire `neg` folder to `data/neg/`

### If you have Parquet files:
- Copy any `.parquet` files directly to `data/` folder
- Especially `shorty_features.parquet` if you have it!

## After Uploading

Once you've copied the files, let me know and I'll:
1. Check what files you have
2. Verify the structure
3. Run the appropriate scripts


