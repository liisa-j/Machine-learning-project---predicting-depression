# Guide: Uploading Your Data Files

## 📁 Expected Data Structure

The project expects data in one of these formats:

### Option 1: Raw JSON Files (for full pipeline)
If you have the original Twitter dataset files, place them here:

```
data/
├─ depression/
│  └─ media/data_dump/udit/submission/dataset/depression/cleaned_data_depression/
│     └─ [user folders, each containing: user.json, anchor_tweet.json, tweets.json]
├─ neg/
│  └─ media/data_dump/udit/submission/dataset/neg/cleaned_data_neg/
│     └─ [user folders, each containing: tweets.json]
```

### Option 2: Already Processed Parquet Files (skip preprocessing)
If you already have processed data files, place them directly in the `data/` folder:

- `data/shorty_features.parquet` - **This is what the models need!**
  OR any of these intermediate files:
  - `data/shorty.parquet`
  - `data/shorty_clean.parquet`
  - `data/tweets.parquet` (positive class)
  - `data/neg_combined.parquet` (negative class)

## 🚀 Quick Start Based on What You Have

### If you have `shorty_features.parquet`:
✅ **You're ready!** Just run any model:
```bash
python src/linear_regression.py
```

### If you have raw JSON files:
1. Extract/copy them to the structure above
2. Run the preprocessing pipeline (see RUN_PROJECT.md)

### If you have zip files:
1. Extract them to `data/` folder
2. The structure should match Option 1 above
3. Run the preprocessing pipeline

## 📝 Steps to Upload

1. **Copy your data files** to the project's `data/` folder
2. **Ensure correct folder structure** (see above)
3. **Let me know what files you have** and I'll help you run the appropriate scripts!


