import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from model_training import load_and_preprocess_data, prepare_training_data

def perform_grid_search(X_train, y_train):
    """
    Perform grid search for hyperparameter tuning
    """
    print("\n=== Performing Grid Search ===")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    # Perform grid search
    print("\nSearching for best parameters...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("\nBest cross-validation score:", grid_search.best_score_)
    
    # Get feature importance from best model
    feature_importance = pd.DataFrame({
        'feature': ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                   'Temperature_diff', 'Power'],
        'importance': grid_search.best_estimator_.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nFeature Importance from best model:")
    print(feature_importance)
    
    return grid_search.best_estimator_

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(X, y)
    
    # Perform grid search
    best_model = perform_grid_search(X_train, y_train)
    
    # Save best model
    print("\n=== Saving Best Model ===")
    joblib.dump(best_model, './models/random_forest_model_tuned.pkl')
    joblib.dump(scaler, './models/scaler_tuned.pkl')
    print("Best model and scaler saved in 'models' directory")
    
    print("\nHyperparameter tuning complete!")

if __name__ == "__main__":
    main() 