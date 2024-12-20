import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_and_explore_data():
    """
    Load and perform initial exploration of the dataset
    """
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('../data/ai4i2020.csv')
    
    # Basic information
    print("\n=== Basic Information ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Check missing values
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.any() else "No missing values found")
    
    # Check duplicates
    print("\n=== Duplicate Rows ===")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    return df

def analyze_target_variable(df):
    """
    Analyze the target variable (Machine failure)
    """
    print("\n=== Target Variable Analysis ===")
    failure_counts = df['Machine failure'].value_counts()
    failure_percentages = (failure_counts / len(df) * 100).round(2)
    
    print("\nFailure Distribution:")
    for value, count in failure_counts.items():
        print(f"Class {value}: {count} samples ({failure_percentages[value]}%)")
    
    # Plot failure distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Machine failure')
    plt.title('Distribution of Machine Failures')
    plt.savefig('../output/failure_distribution.png')
    plt.close()

def analyze_numeric_features(df):
    """
    Analyze numeric features
    """
    print("\n=== Numeric Features Analysis ===")
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df[numeric_cols].describe())
    
    # Correlation analysis
    correlation_matrix = df[numeric_cols + ['Machine failure']].corr()
    print("\nCorrelation with Machine failure:")
    print(correlation_matrix['Machine failure'].sort_values(ascending=False))
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../output/correlation_matrix.png')
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, hue='Machine failure', multiple="stack", ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.savefig('../output/feature_distributions.png')
    plt.close()

def analyze_failure_types(df):
    """
    Analyze different types of failures
    """
    print("\n=== Failure Types Analysis ===")
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    for failure_type in failure_types:
        count = df[failure_type].sum()
        percentage = (count / len(df) * 100).round(2)
        print(f"{failure_type}: {count} occurrences ({percentage}%)")

def main():
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('../output'):
        os.makedirs('../output')
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Perform analyses
    analyze_target_variable(df)
    analyze_numeric_features(df)
    analyze_failure_types(df)
    
    print("\nAnalysis complete! Check the 'output' directory for visualizations.")

if __name__ == "__main__":
    main() 