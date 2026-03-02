import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _engineer_features(X):
    """Create interaction / ratio features that capture player behaviour patterns."""
    Xe = X.copy()

    Xe['PlayTimePerSession'] = Xe['PlayTimeHours'] / Xe['SessionsPerWeek'].replace(0, np.nan)
    Xe['PlayTimePerSession'] = Xe['PlayTimePerSession'].fillna(0)

    Xe['AchievementsPerLevel'] = Xe['AchievementsUnlocked'] / Xe['PlayerLevel'].replace(0, np.nan)
    Xe['AchievementsPerLevel'] = Xe['AchievementsPerLevel'].fillna(0)
    
    Xe['TotalWeeklyMinutes'] = Xe['SessionsPerWeek'] * Xe['AvgSessionDurationMinutes']
    return Xe


ENGINEERED_NUM = ['PlayTimePerSession', 'AchievementsPerLevel', 'TotalWeeklyMinutes']


def fit_preprocessors(X_train, numeric_features, categorical_features):
    X_train_e = _engineer_features(X_train)
    all_numeric = numeric_features + ENGINEERED_NUM

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    scaler.fit(X_train_e[all_numeric])
    encoder.fit(X_train_e[categorical_features])

    return scaler, encoder


def apply_preprocessors(X, scaler, encoder, numeric_features, categorical_features):
    Xe = _engineer_features(X)
    all_numeric = numeric_features + ENGINEERED_NUM

    X_num = scaler.transform(Xe[all_numeric])
    X_num_df = pd.DataFrame(X_num, columns=all_numeric, index=X.index)

    X_cat = encoder.transform(Xe[categorical_features])
    cat_columns = encoder.get_feature_names_out(categorical_features)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_columns, index=X.index)

    return pd.concat([X_num_df, X_cat_df], axis=1)