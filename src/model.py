from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

class DiagnosticModel:
    def __init__(self):
        # Baseline: Constrained depth to simulate a conservative "out-of-box" model
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        # Comparison: manually specified higher-complexity configuration; not a tuning search
        self.tuned_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            max_features='sqrt',
            random_state=42
        )

    @staticmethod
    def split(X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_and_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = self.split(X, y)

        # 1. Train Baseline
        self.base_model.fit(X_train, y_train)
        base_preds = self.base_model.predict(X_test)
        base_acc = accuracy_score(y_test, base_preds)

        # 2. Train Tuned
        self.tuned_model.fit(X_train, y_train)
        tuned_preds = self.tuned_model.predict(X_test)
        tuned_acc = accuracy_score(y_test, tuned_preds)

        return base_acc, tuned_acc

    def detailed_report(self, X, y):
        """Fits both models and returns test-set metrics plus 5-fold CV accuracy on the training split."""
        X_train, X_test, y_train, y_test = self.split(X, y)
        report = {}
        for name, model in (('baseline', self.base_model), ('comparison', self.tuned_model)):
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            report[name] = {
                'accuracy': accuracy_score(y_test, preds),
                'precision': precision_score(y_test, preds),
                'recall': recall_score(y_test, preds),
                'f1': f1_score(y_test, preds),
                'roc_auc': roc_auc_score(y_test, proba),
                'cv_accuracy_mean': cv.mean(),
                'cv_accuracy_std': cv.std(),
                'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
                'feature_importance': dict(zip(X.columns, model.feature_importances_.round(4).tolist())),
            }
        return report
