from sklearn.ensemble import RandomForestClassifier

def train_model(X_train_processed, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_processed, y_train)
    
    return model