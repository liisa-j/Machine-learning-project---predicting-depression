"""
Create a formatted comparison table from all model results
"""
import pandas as pd
import json
from pathlib import Path

print("="*70)
print("MODEL COMPARISON TABLE")
print("="*70)

# Load baseline results
baseline_file = Path("baseline_models_results.json")
if baseline_file.exists():
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)
else:
    baseline_results = {}

# Transformer results (from summary file)
transformer_results = {
    'Transformer': {
        'Accuracy': 0.9885,
        'F1 Score': 0.9885,
        'Training Time (min)': 59.50
    }
}

# Combine all results
all_results = {**transformer_results, **baseline_results}

# Create DataFrame
df = pd.DataFrame(all_results).T
df = df.sort_values('Accuracy', ascending=False)

# Format for display
df_display = df.copy()
df_display['Accuracy'] = df_display['Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
df_display['F1 Score'] = df_display['F1 Score'].apply(lambda x: f"{x:.4f}")
df_display['Training Time (min)'] = df_display['Training Time (min)'].apply(lambda x: f"{x:.2f}")

print("\n" + df_display.to_string())
print("\n" + "="*70)

# Save to CSV
df.to_csv("model_comparison_table.csv")
print(f"\n✓ Comparison table saved to: model_comparison_table.csv")

# Create summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nBest Model: {df.index[0]}")
print(f"  Accuracy: {df.loc[df.index[0], 'Accuracy']:.4f} ({df.loc[df.index[0], 'Accuracy']*100:.2f}%)")
print(f"  F1 Score: {df.loc[df.index[0], 'F1 Score']:.4f}")

if len(df) > 1:
    second_best = df.index[1]
    improvement = (df.loc[df.index[0], 'Accuracy'] - df.loc[second_best, 'Accuracy']) * 100
    print(f"\nImprovement over {second_best}: +{improvement:.2f}% accuracy")

print("\n" + "="*70)

