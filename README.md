# Healthcare Diagnostic Hub 🏥

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

A predictive analytics prototype designed to assist early disease risk identification using structured patient data.  
This project demonstrates **model optimization and performance benchmarking** using Random Forest classifiers on synthetic medical records.

## ⚡ Business Objective
* **Goal:** Improve early disease detection accuracy to support clinical decision-making.
* **Metric:** Reduction of false negatives through hyperparameter tuning.
* **Impact:** Demonstrated a measurable performance gain over baseline heuristics.

## 🧠 Technical Approach
* **Data:** Synthetic dataset mimicking the feature distribution of the Pima Indians Diabetes dataset.
* **Model:** Random Forest Classifier (Ensemble Learning).
* **Optimization:** Comparison of a standard baseline vs. a hyperparameter-tuned model.

## 📊 Performance Summary
Statistical benchmarking was conducted on a held-out test set.

| Model Configuration | Accuracy |
| :--- | :--- |
| Baseline Random Forest | ~71% |
| **Optimized Random Forest** | **~86%** |
| **Relative Improvement** | **~21%** |

*> **Note:** "Relative Improvement" is calculated as `(Optimized - Baseline) / Baseline`. Results confirm that parameter tuning significantly enhances predictive power.*

## 📂 Repository Structure
* `src/model.py`: Random Forest training logic and configuration.
* `src/data_gen.py`: Synthetic patient data generator.
* `tuning_analysis.py`: Benchmarking script to validate accuracy gains.

## 🚀 How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the benchmarking script:
   `python tuning_analysis.py`
