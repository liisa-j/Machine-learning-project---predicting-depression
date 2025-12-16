# Current Datasets Summary

## 📊 Available Datasets

### ✅ **Reddit Datasets (Ready to Use)**

#### 1. **reddit_features.parquet** (23.47 MB) ⭐ **MAIN DATASET FOR MODELS**
- **Rows:** 43,373
- **Columns:** 59
- **Purpose:** Feature-extracted dataset ready for machine learning
- **Features include:**
  - Basic text metrics (char_count, word_count, sentence_count)
  - Linguistic features (uppercase_ratio, elongation_count, punctuation counts)
  - Sentiment analysis (TextBlob polarity/subjectivity, VADER scores)
  - Empath categories (sadness, fear, anger, health, social, etc.)
  - Readability scores (Flesch reading ease, Flesch-Kincaid grade)
  - First person pronouns, negations, function words
- **Labels:** 
  - Class 0 (Not depressed): 22,164 (51.1%)
  - Class 1 (Depressed): 21,209 (48.9%)
- **Status:** ✅ Ready for model training

#### 2. **reddit_full_dataset.parquet** (26.26 MB)
- **Rows:** 43,373
- **Columns:** 96
- **Purpose:** Complete dataset with all original features
- **Includes:**
  - All features from reddit_features.parquet
  - Additional LIWC (Linguistic Inquiry Word Count) features
  - Readability indices (automated_readability_index, coleman_liau_index, etc.)
  - Stress indicators (economic_stress, isolation, substance_use, etc.)
  - Absolutist word counts
- **Status:** ✅ Available for analysis

#### 3. **reddit_text_classification_dataset.parquet** (21.00 MB)
- **Rows:** 43,373
- **Columns:** 3 (author, clean_text, label_encoded)
- **Purpose:** Minimal text classification dataset
- **Status:** ✅ Used as input for feature extraction

### 📥 **Source CSV Files**

#### 4. **depression_pre_features_tfidf_256.csv** (57.68 MB)
- **Rows:** ~14,346 (estimated)
- **Columns:** 350
- **Source:** r/depression subreddit posts
- **Features:** 
  - 256 TF-IDF features
  - LIWC features
  - Readability metrics
  - Sentiment scores
- **Status:** ✅ Source data (already processed into parquet)

#### 5. **fitness_pre_features_tfidf_256.csv** (52.06 MB)
- **Rows:** ~13,986 (estimated)
- **Columns:** 350
- **Source:** r/fitness subreddit posts (control group)
- **Features:** Same as depression CSV
- **Status:** ✅ Source data (already processed into parquet)

---

## ❌ **Missing Datasets**

### Twitter Datasets
- **depression.zip** - Not downloaded
- **neg.zip** - Not downloaded
- **Raw Twitter JSON files** - Not available

These would be needed to run the Twitter-based models (RF.py, linear_regression.py, svm.py for Twitter data).

---

## 🎯 **What You Can Do Now**

### ✅ **Ready to Run:**
1. **Reddit Logistic Regression** - `python src/reddit_LR.py` ✅ (Already run - 89.1% accuracy)
2. **Other Reddit models** - Can be adapted to use `reddit_features.parquet`

### ⚠️ **Cannot Run Yet:**
- Twitter-based models (need Twitter dataset download)
- Models expecting `data/shorty_features.parquet`

---

## 📈 **Dataset Statistics**

- **Total datasets:** 5 files
- **Total size:** ~160 MB
- **Total rows (processed):** 43,373 Reddit posts
- **Class balance:** ~51% vs 49% (well balanced)
- **Feature types:** Linguistic, sentiment, psychological, readability

---

## 🔄 **Data Pipeline Status**

```
✅ CSV Files (Source)
   ↓
✅ create_parquets.py (Processed)
   ↓
✅ reddit_text_classification_dataset.parquet
   ↓
✅ features_from_reddit_dataset.py (Feature Extraction)
   ↓
✅ reddit_features.parquet (Ready for Models)
   ↓
✅ reddit_LR.py (Model Training - 89.1% accuracy)
```

---

## 📝 **Quick Reference**

**Main dataset for models:** `data/reddit_features.parquet`
- 43,373 rows
- 59 features
- Binary classification (depressed vs not depressed)
- Balanced classes


