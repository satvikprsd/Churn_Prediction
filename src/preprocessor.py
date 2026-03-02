import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def fit_preprocessors(X_train, numeric_features, categorical_features):
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    scaler.fit(X_train[numeric_features])
    encoder.fit(X_train[categorical_features])
    
    return scaler, encoder

def apply_preprocessors(X, scaler, encoder, numeric_features, categorical_features):
    X_num = scaler.transform(X[numeric_features])
    X_num_df = pd.DataFrame(X_num, columns=numeric_features, index=X.index)
    
    X_cat = encoder.transform(X[categorical_features])
    cat_columns = encoder.get_feature_names_out(categorical_features)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_columns, index=X.index)
    
    return pd.concat([X_num_df, X_cat_df], axis=1)