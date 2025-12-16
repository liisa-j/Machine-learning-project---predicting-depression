"""
Create visualizations for model comparison and results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("Creating visualizations...")

# Load results
baseline_file = Path("baseline_models_results.json")
if baseline_file.exists():
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)
else:
    baseline_results = {}

transformer_results = {
    'Transformer': {
        'Accuracy': 0.9885,
        'F1 Score': 0.9885,
        'Training Time (min)': 59.50
    }
}

all_results = {**transformer_results, **baseline_results}

# ========== 1. Performance Comparison Bar Chart ==========
print("1. Creating performance comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = list(all_results.keys())
accuracies = [all_results[m]['Accuracy'] for m in models]
f1_scores = [all_results[m]['F1 Score'] for m in models]

# Accuracy comparison
axes[0].bar(models, accuracies, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1.1])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[0].text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontweight='bold')

# F1 Score comparison
axes[1].bar(models, f1_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1.1])
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (model, f1) in enumerate(zip(models, f1_scores)):
    axes[1].text(i, f1 + 0.02, f'{f1:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: model_performance_comparison.png")
plt.close()

# ========== 2. Confusion Matrix Heatmap for Transformer ==========
print("2. Creating confusion matrix heatmap...")
transformer_cm = np.array([[1977, 23], [23, 1977]])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'],
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title('Transformer Model - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: transformer_confusion_matrix.png")
plt.close()

# ========== 3. Training Time vs Accuracy Trade-off ==========
print("3. Creating training time vs accuracy chart...")
fig, ax = plt.subplots(figsize=(10, 6))

times = [all_results[m]['Training Time (min)'] for m in models]
colors_map = {'Transformer': '#2ecc71', 'Logistic Regression': '#3498db', 
              'Random Forest': '#9b59b6', 'SVM': '#e74c3c'}

for i, model in enumerate(models):
    ax.scatter(times[i], accuracies[i], s=300, color=colors_map.get(model, '#95a5a6'),
               alpha=0.7, edgecolors='black', linewidth=2)
    ax.annotate(model, (times[i], accuracies[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Training Time vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('training_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: training_time_vs_accuracy.png")
plt.close()

# ========== 4. Class Distribution Chart ==========
print("4. Creating class distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dataset class distribution
class_counts = [10000, 10000]
class_labels = ['Not Depressed', 'Depressed']
colors = ['#3498db', '#e74c3c']

axes[0].bar(class_labels, class_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[0].set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(class_labels, class_counts)):
    axes[0].text(i, count + 200, str(count), ha='center', fontweight='bold', fontsize=11)

# Pie chart
axes[1].pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: class_distribution.png")
plt.close()

# ========== 5. Error Rate Comparison ==========
print("5. Creating error rate comparison...")
error_rates = {
    'Transformer': 46 / 4000 * 100,
    'Logistic Regression': 458 / 4000 * 100,
    'Random Forest': 501 / 4000 * 100,
    'SVM': 1908 / 4000 * 100
}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(list(error_rates.keys()), list(error_rates.values()), 
              color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], 
              alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Error Rate Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, (model, rate) in zip(bars, error_rates.items()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{rate:.2f}%', ha='center', fontweight='bold', fontsize=10)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('error_rate_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: error_rate_comparison.png")
plt.close()

# ========== 6. Comprehensive Comparison Dashboard ==========
print("6. Creating comprehensive dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Accuracy comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(models, accuracies, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy', fontweight='bold', fontsize=12)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)
for i, acc in enumerate(accuracies):
    ax1.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=9, fontweight='bold')

# F1 Score
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(models, f1_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax2.set_ylabel('F1 Score', fontweight='bold')
ax2.set_title('F1 Score', fontweight='bold', fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)
for i, f1 in enumerate(f1_scores):
    ax2.text(i, f1 + 0.02, f'{f1:.2%}', ha='center', fontsize=9, fontweight='bold')

# Training Time
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(models, times, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax3.set_ylabel('Time (min)', fontweight='bold')
ax3.set_title('Training Time', fontweight='bold', fontsize=12)
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3)
for i, t in enumerate(times):
    ax3.text(i, t * 1.5, f'{t:.2f}', ha='center', fontsize=9, fontweight='bold')

# Error Rate
ax4 = fig.add_subplot(gs[1, 0])
error_values = list(error_rates.values())
ax4.bar(models, error_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax4.set_ylabel('Error Rate (%)', fontweight='bold')
ax4.set_title('Error Rate', fontweight='bold', fontsize=12)
ax4.grid(axis='y', alpha=0.3)
for i, err in enumerate(error_values):
    ax4.text(i, err + 1, f'{err:.2f}%', ha='center', fontsize=9, fontweight='bold')

# Confusion Matrix
ax5 = fig.add_subplot(gs[1, 1:])
sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'],
            cbar_kws={'label': 'Count'}, ax=ax5)
ax5.set_xlabel('Predicted', fontweight='bold')
ax5.set_ylabel('Actual', fontweight='bold')
ax5.set_title('Transformer Confusion Matrix', fontweight='bold', fontsize=12)

# Class Distribution
ax6 = fig.add_subplot(gs[2, 0])
ax6.bar(class_labels, class_counts, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Count', fontweight='bold')
ax6.set_title('Class Distribution', fontweight='bold', fontsize=12)
ax6.grid(axis='y', alpha=0.3)

# Time vs Accuracy
ax7 = fig.add_subplot(gs[2, 1:])
for i, model in enumerate(models):
    ax7.scatter(times[i], accuracies[i], s=400, color=colors_map.get(model, '#95a5a6'),
               alpha=0.7, edgecolors='black', linewidth=2)
    ax7.annotate(model, (times[i], accuracies[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
ax7.set_xlabel('Training Time (min, log scale)', fontweight='bold')
ax7.set_ylabel('Accuracy', fontweight='bold')
ax7.set_title('Performance vs Training Time', fontweight='bold', fontsize=12)
ax7.set_xscale('log')
ax7.grid(True, alpha=0.3)

plt.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: model_comparison_dashboard.png")
plt.close()

print("\n" + "="*70)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
print("="*70)
print("\nGenerated files:")
print("  1. model_performance_comparison.png")
print("  2. transformer_confusion_matrix.png")
print("  3. training_time_vs_accuracy.png")
print("  4. class_distribution.png")
print("  5. error_rate_comparison.png")
print("  6. model_comparison_dashboard.png")
print("\n" + "="*70)

