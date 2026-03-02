import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_model(model, X_test_processed, y_test):
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }
    return metrics

def get_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return importances.head(5)