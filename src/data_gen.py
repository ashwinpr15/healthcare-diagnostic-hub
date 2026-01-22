import pandas as pd
import numpy as np

def generate_patient_data(n_samples=1000):
    """
    Generates synthetic patient data mimicking Pima Diabetes structure.
    Features: Glucose, BMI, Age, BloodPressure, Insulin.
    Target: Outcome (0 = No Diabetes, 1 = Diabetes)
    """
    np.random.seed(42)

    # 1. Generate Feature Data
    data = {
        'Glucose': np.random.normal(120, 30, n_samples),
        'BMI': np.random.normal(32, 6, n_samples),
        'Age': np.random.randint(21, 80, n_samples),
        'BloodPressure': np.random.normal(70, 10, n_samples),
        'Insulin': np.random.normal(80, 20, n_samples)
    }

    df = pd.DataFrame(data)

    # 2. Create Target Variable (Outcome)
    # We add significant noise to prevent the model from getting 100% accuracy
    risk_score = (df['Glucose'] * 0.6) + (df['BMI'] * 1.5) + (df['Age'] * 0.2)
    threshold = risk_score.median()

    # Noise injection to simulate real-world medical data ambiguity
    noise = np.random.normal(0, 15, n_samples)
    df['Outcome'] = (risk_score + noise > threshold).astype(int)

    return df
