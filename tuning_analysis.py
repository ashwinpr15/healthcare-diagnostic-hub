import json
import sys
from pathlib import Path

from src.data_gen import generate_patient_data
from src.model import DiagnosticModel

print("--- Healthcare Diagnostic Hub ---")
print("Loading Patient Data...")

# 1. Load Data
df = generate_patient_data(n_samples=2000)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Dataset Shape: {df.shape}")
print("Training Models (Baseline vs. Tuned)...")

# 2. Run Comparison
model_system = DiagnosticModel()
base_acc, tuned_acc = model_system.train_and_evaluate(X, y)

# 3. Calculate Relative Improvement
# Formula: (New - Old) / Old
rel_improvement = (tuned_acc - base_acc) / base_acc

# 4. Print Report
print("\n" + "="*40)
print(f" PERFORMANCE BENCHMARK")
print("="*40)
print(f"Baseline Model Accuracy:   {base_acc*100:.1f}%")
print(f"Comparison Model Accuracy:  {tuned_acc*100:.1f}%")
print("-" * 40)
print(f"RELATIVE CHANGE:           {rel_improvement*100:+.1f}%")
print("="*40)

# 5. Fuller report: precision, recall, ROC AUC, cross-validation, plots
if '--quick' not in sys.argv:
    report = model_system.detailed_report(X, y)
    print(f"\n{'Metric':<18}{'Baseline':>10}{'Comparison':>12}")
    for key in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_accuracy_mean'):
        print(f"{key:<18}{report['baseline'][key]:>10.3f}{report['comparison'][key]:>12.3f}")

    out = Path('results')
    out.mkdir(exist_ok=True)
    report['data'] = {'samples': len(df), 'positive_rate': round(float(y.mean()), 4), 'test_size': 0.2, 'seed': 42}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ('baseline', 'comparison')):
        cm = report[name]['confusion_matrix']
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha='center', va='center', fontsize=14,
                        color='white' if cm[i][j] > max(map(max, cm)) / 2 else 'black')
        ax.set_xticks([0, 1], ['Predicted 0', 'Predicted 1'])
        ax.set_yticks([0, 1], ['Actual 0', 'Actual 1'])
        ax.set_title(f"{name.title()} (accuracy {report[name]['accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrices.png', dpi=120)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(report['baseline']['feature_importance'])
    base = [report['baseline']['feature_importance'][n] for n in names]
    comp = [report['comparison']['feature_importance'][n] for n in names]
    idx = range(len(names))
    ax.barh([i + 0.2 for i in idx], base, height=0.4, label='Baseline', color='#7fb3d5')
    ax.barh([i - 0.2 for i in idx], comp, height=0.4, label='Comparison', color='#1f618d')
    ax.set_yticks(list(idx), names)
    ax.set_xlabel('Feature importance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'feature_importance.png', dpi=120)
    print("\nSaved results/metrics.json, results/confusion_matrices.png and results/feature_importance.png")
