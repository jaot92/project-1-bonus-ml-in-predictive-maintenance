import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_output_dir():
    """Create output directory for feature engineering results"""
    if not os.path.exists('./output/feature_engineering'):
        os.makedirs('./output/feature_engineering')

def load_data():
    """Load the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('./data/ai4i2020.csv')
    return df

def create_rolling_features(df, window_sizes=[3, 5, 10]):
    """Create rolling average features"""
    print("\nCreating rolling average features...")
    
    # Sort by Product ID and Type to maintain logical sequence
    df = df.sort_values(['Product ID', 'Type'])
    
    # Features to calculate rolling averages for
    features = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Calculate rolling averages for each window size
    for window in window_sizes:
        for feature in features:
            col_name = f'{feature}_rolling_{window}'
            df[col_name] = df.groupby('Type')[feature].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
    
    return df

def create_interaction_features(df):
    """Create interaction features"""
    print("Creating interaction features...")
    
    # Temperature related
    df['Temperature_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Temperature_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']
    
    # Power and efficiency related
    df['Power'] = (df['Rotational speed [rpm]'] * df['Torque [Nm]']) / 9550  # Power in kW
    df['Power_per_temp'] = df['Power'] / df['Process temperature [K]']
    df['Efficiency'] = df['Power'] / (df['Tool wear [min]'] + 1)  # Adding 1 to avoid division by zero
    
    # Wear rate
    df['Wear_rate'] = df['Tool wear [min]'] / df.groupby('Product ID').cumcount().add(1)
    
    return df

def perform_feature_selection(X, y, n_features_to_select=10):
    """Perform recursive feature elimination"""
    print("\nPerforming recursive feature elimination...")
    
    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize RFE
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select)
    
    # Fit RFE
    rfe = rfe.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Get feature ranking
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    return selected_features, feature_ranking

def plot_feature_importance(feature_ranking):
    """Plot feature importance ranking"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_ranking.head(15), x='Rank', y='Feature')
    plt.title('Top 15 Features by RFE Ranking')
    plt.tight_layout()
    plt.savefig('./output/feature_engineering/feature_ranking.png')
    plt.close()

def main():
    # Create output directory
    create_output_dir()
    
    # Load data
    df = load_data()
    
    # Create rolling average features
    df = create_rolling_features(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Prepare features for selection
    feature_cols = [col for col in df.columns if col not in ['Product ID', 'Type', 'Machine failure', 
                                                            'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    X = df[feature_cols]
    y = df['Machine failure']
    
    # Perform feature selection
    selected_features, feature_ranking = perform_feature_selection(X, y)
    
    # Plot feature importance
    plot_feature_importance(feature_ranking)
    
    # Save selected features
    pd.DataFrame(selected_features, columns=['Feature']).to_csv(
        './output/feature_engineering/selected_features.csv', index=False
    )
    
    # Save engineered dataset
    df.to_csv('./data/engineered_features.csv', index=False)
    
    print("\nFeature engineering complete!")
    print(f"\nSelected top features:\n{', '.join(selected_features)}")

if __name__ == "__main__":
    main() 