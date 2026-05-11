from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import pandas as pd


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"Accuracy : {results[name]['accuracy']:.4f}")
        print(f"ROC-AUC  : {results[name]['roc_auc']:.4f}")
        print(results[name]['report'])

    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\nBest Model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

    rf = results['Random Forest']['model']
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importances:\n{importance_df}")

    return results, best_name, importance_df
