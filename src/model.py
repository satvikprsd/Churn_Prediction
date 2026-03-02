from sklearn.ensemble import RandomForestClassifier

def train_model(X_train_processed, y_train):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced'
    )
    model.fit(X_train_processed, y_train)

    return model