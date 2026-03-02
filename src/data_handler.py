import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath):
    df = pd.read_csv(filepath)
    
    df['Churn'] = df['EngagementLevel'].apply(lambda x: 1 if x == 'Low' else 0)

    df.drop(columns=['EngagementLevel'], inplace=True)
    df.drop(columns=['PlayerID'], inplace=True)
    df.dropna(inplace=True)
    
    numeric_features = ['PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']
    categorical_features = ['GameDifficulty', 'Gender'] 
    
    X = df[numeric_features + categorical_features]
    y = df['Churn']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), numeric_features, categorical_features