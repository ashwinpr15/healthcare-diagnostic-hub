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
print(f"Optimized Model Accuracy:  {tuned_acc*100:.1f}%")
print("-" * 40)
print(f"RELATIVE IMPROVEMENT:      +{rel_improvement*100:.1f}%")
print("="*40)
