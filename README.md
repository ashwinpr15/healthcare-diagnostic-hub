<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner-light.svg" alt="Synthetic classification benchmark" width="100%">
</picture>

# Synthetic classification benchmark

Educational comparison of two Random Forest configurations using generated structured records. The five features are Glucose, BMI, Age, BloodPressure, and Insulin. A noisy arithmetic rule creates the binary label.

## Implementation

- `src/data_gen.py`: creates synthetic features and constructed labels with seed 42.
- `src/model.py`: compares a depth-limited 100-tree model with a manually configured 300-tree model on one stratified 80/20 split.
- `tuning_analysis.py`: prints accuracy and signed relative change. A comparison can be negative; improvement is not assumed.

## Run from the repository root

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.txt
python tuning_analysis.py
```

## Evaluation scope

These are fixed configurations, not the result of a hyperparameter search. Accuracy on constructed labels does not measure disease prediction in patients. The code includes no real Pima data, cancer model, SMOTE, explainability, false-negative comparison, or clinical validation.

For a stronger future experiment, select parameters using cross-validation on training data, reserve an untouched test set, report precision/recall and confusion matrices, and compare a simple baseline. Record exact package versions with results. This is a learning demonstration, not a diagnostic tool.

## Targeted checks

Run `python test_evaluation.py` from the repository root after installing dependencies.

## License

Released under the [MIT License](LICENSE).
