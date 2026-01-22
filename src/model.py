from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DiagnosticModel:
    def __init__(self):
        # Baseline: Constrained depth to simulate a conservative "out-of-box" model
        self.base_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )

        # Tuned: Optimized parameters for higher complexity and better fit
        self.tuned_model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=None, 
            min_samples_split=5, 
            max_features='sqrt',
            random_state=42
        )

    def train_and_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Train Baseline
        self.base_model.fit(X_train, y_train)
        base_preds = self.base_model.predict(X_test)
        base_acc = accuracy_score(y_test, base_preds)

        # 2. Train Tuned
        self.tuned_model.fit(X_train, y_train)
        tuned_preds = self.tuned_model.predict(X_test)
        tuned_acc = accuracy_score(y_test, tuned_preds)

        return base_acc, tuned_acc
