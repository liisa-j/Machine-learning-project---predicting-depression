# Model Comparison Report: Transformer vs Baseline Models

## Executive Summary

This report compares the performance of a **DistilBERT Transformer** model against three baseline machine learning models (Logistic Regression, Random Forest, and SVM) for depression classification on Reddit posts.

---

## Dataset

- **Source:** Reddit posts (r/depression vs r/fitness)
- **Total Samples:** 20,000 (balanced: 10,000 per class)
- **Train/Val/Test Split:** 14,400 / 1,600 / 4,000
- **Features:** 
  - Transformer: Raw text (max 128 tokens)
  - Baselines: 56 engineered features (linguistic, sentiment, readability)

---

## Model Results Comparison

| Model | Accuracy | F1 Score | Training Time | Best For |
|-------|----------|----------|---------------|----------|
| **DistilBERT (Transformer)** | **98.85%** | **98.85%** | 59.50 min | **Best Performance** |
| Logistic Regression | 88.55% | 88.55% | <0.01 min | Fast Training |
| Random Forest | 87.48% | 87.47% | 0.01 min | Fast Training |
| SVM | 52.30% | 41.85% | 0.01 min | Not Recommended |

---

## Detailed Performance Metrics

### 1. DistilBERT Transformer ⭐

**Configuration:**
- Base Model: DistilBERT-base-uncased
- Max Sequence Length: 128 tokens
- Epochs: 2
- Batch Size: 16 (train), 32 (eval)

**Results:**
- **Accuracy:** 98.85%
- **F1 Score (Weighted):** 98.85%
- **F1 Score (Macro):** 98.85%
- **Training Time:** 59.50 minutes

**Confusion Matrix:**
```
                Predicted
Actual          Not Depressed  Depressed
Not Depressed       1977          23
Depressed            23         1977
```

**Strengths:**
- Highest accuracy (98.85%)
- Excellent balance across both classes
- Only 46 misclassifications out of 4,000
- Captures complex language patterns

**Weaknesses:**
- Longest training time (59.5 minutes)
- Requires GPU for faster training
- More complex to deploy

---

### 2. Logistic Regression

**Configuration:**
- Features: 56 engineered features
- Regularization: L2
- Class Weight: Balanced

**Results:**
- **Accuracy:** 88.55%
- **F1 Score (Weighted):** 88.55%
- **Training Time:** <0.01 minutes

**Confusion Matrix:**
```
                Predicted
Actual          Not Depressed  Depressed
Not Depressed       1808         192
Depressed            266        1734
```

**Strengths:**
- Very fast training (<1 second)
- Interpretable (feature coefficients)
- Good baseline performance (88.55%)
- Easy to deploy

**Weaknesses:**
- 10.3% lower accuracy than transformer
- 458 misclassifications (vs 46 for transformer)
- Requires feature engineering

---

### 3. Random Forest

**Configuration:**
- Estimators: 100 trees
- Max Depth: 20
- Class Weight: Balanced

**Results:**
- **Accuracy:** 87.48%
- **F1 Score (Weighted):** 87.47%
- **Training Time:** 0.01 minutes

**Confusion Matrix:**
```
                Predicted
Actual          Not Depressed  Depressed
Not Depressed       1753         247
Depressed            254        1746
```

**Strengths:**
- Fast training (~1 second)
- Handles non-linear relationships
- Feature importance available
- Good performance (87.48%)

**Weaknesses:**
- 11.37% lower accuracy than transformer
- 501 misclassifications
- Less interpretable than Logistic Regression

---

### 4. Support Vector Machine (SVM)

**Configuration:**
- Kernel: Linear
- Class Weight: Balanced
- Max Iterations: 1000

**Results:**
- **Accuracy:** 52.30%
- **F1 Score (Weighted):** 41.85%
- **Training Time:** 0.01 minutes

**Confusion Matrix:**
```
                Predicted
Actual          Not Depressed  Depressed
Not Depressed        198        1802
Depressed            106        1894
```

**Strengths:**
- Very fast training
- Memory efficient

**Weaknesses:**
- Poor performance (52.30% - barely better than random)
- Did not converge properly
- High false positive rate
- **Not recommended for this task**

---

## Performance Analysis

### Accuracy Comparison

```
Transformer:        ████████████████████ 98.85%
Logistic Regression: ████████████████░░░░ 88.55%
Random Forest:       ███████████████░░░░░ 87.48%
SVM:                 ██████░░░░░░░░░░░░░ 52.30%
```

### Improvement Over Baselines

| Baseline Model | Transformer Improvement |
|----------------|------------------------|
| Logistic Regression | **+10.30%** accuracy |
| Random Forest | **+11.37%** accuracy |
| SVM | **+46.55%** accuracy |

### Training Time Comparison

| Model | Training Time | Speed vs Transformer |
|-------|---------------|----------------------|
| Transformer | 59.50 min | 1x (baseline) |
| Logistic Regression | <0.01 min | ~3,570x faster |
| Random Forest | 0.01 min | ~3,570x faster |
| SVM | 0.01 min | ~3,570x faster |

---

## Key Findings

### 1. **Transformer Outperforms All Baselines**
   - **10-11% higher accuracy** than Logistic Regression and Random Forest
   - **46.55% higher accuracy** than SVM
   - Achieves near-perfect performance (98.85%)

### 2. **Trade-off: Performance vs Speed**
   - Transformer: Best accuracy but slowest training (59.5 min)
   - Baselines: Lower accuracy but much faster (<1 second)

### 3. **Misclassification Analysis**
   - **Transformer:** 46 errors (1.15% error rate)
   - **Logistic Regression:** 458 errors (11.45% error rate)
   - **Random Forest:** 501 errors (12.53% error rate)
   - **SVM:** 1,908 errors (47.70% error rate)

### 4. **Class Balance**
   - Transformer: Perfectly balanced (23 FP, 23 FN)
   - Logistic Regression: Slight bias (192 FP, 266 FN)
   - Random Forest: Slight bias (247 FP, 254 FN)
   - SVM: Severe bias (1,802 FP, 106 FN)

---

## Recommendations

### When to Use Transformer:
✅ **Recommended for:**
- Production systems requiring highest accuracy
- Research applications
- When training time is not critical
- When you have GPU resources

### When to Use Baselines:
✅ **Logistic Regression recommended for:**
- Real-time applications requiring fast inference
- When interpretability is important
- Prototyping and quick experiments
- Resource-constrained environments

✅ **Random Forest recommended for:**
- When you need feature importance
- Handling non-linear patterns
- Fast training with good performance

❌ **SVM not recommended** for this task due to poor convergence and performance.

---

## Conclusion

The **DistilBERT Transformer model significantly outperforms all baseline models**, achieving **98.85% accuracy** compared to 88.55% (Logistic Regression) and 87.48% (Random Forest). While the transformer requires longer training time, the substantial performance improvement (10-11% accuracy gain) makes it the **recommended choice for production deployment** where accuracy is critical.

For applications requiring faster training or inference, **Logistic Regression** provides a good balance with 88.55% accuracy and near-instantaneous training.

---

## Files Generated

- `transformer_results_summary.txt` - Transformer detailed results
- `baseline_models_results.json` - Baseline models results
- `TRANSFORMER_MODEL_REPORT.md` - Transformer model report
- `MODEL_COMPARISON_REPORT.md` - This comparison report

---

*Report generated from model training results*

