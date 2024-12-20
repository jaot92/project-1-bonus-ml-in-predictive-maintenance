import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess_data():
    """
    Load and preprocess the dataset
    """
    print("\n=== Loading and Preprocessing Data ===")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('../data/ai4i2020.csv')
    
    # Select features
    numeric_features = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]'
    ]
    
    # Create feature matrix X and target vector y
    X = df[numeric_features].copy()
    y = df['Machine failure']
    
    # Feature Engineering
    print("\nPerforming Feature Engineering...")
    X['Temperature_diff'] = X['Process temperature [K]'] - X['Air temperature [K]']
    X['Power'] = (X['Rotational speed [rpm]'] * X['Torque [Nm]']) / 9550  # Power in kW
    
    # Print feature information
    print("\nFeatures used in the model:")
    for col in X.columns:
        print(f"- {col}")
    
    return X, y

def prepare_training_data(X, y):
    """
    Prepare data for training (scaling and handling imbalance)
    """
    print("\n=== Preparing Training Data ===")
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Print class distribution after SMOTE
    print("\nClass distribution after SMOTE:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"Class {value}: {count} samples")
    
    return (X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the Random Forest model
    """
    print("\n=== Training and Evaluating Model ===")
    
    # Initialize and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Average F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X_train.shape[1] * ['placeholder'],  # Will be replaced with actual feature names
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance)
    
    return model

def save_model(model, scaler):
    """
    Save the trained model and scaler
    """
    print("\n=== Saving Model and Scaler ===")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    # Save model and scaler
    joblib.dump(model, './models/random_forest_model.pkl')
    joblib.dump(scaler, './models/scaler.pkl')
    print("Model and scaler saved in 'models' directory")

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(X, y)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main() 