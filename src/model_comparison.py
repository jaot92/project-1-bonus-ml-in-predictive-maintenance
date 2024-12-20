import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from model_training import load_and_preprocess_data, prepare_training_data
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_dir():
    """Create output directory for model comparison results"""
    if not os.path.exists('./output/model_comparison'):
        os.makedirs('./output/model_comparison')

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    print(f"\n=== Training {model_name} ===")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Average F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'./output/model_comparison/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std(),
        'model': model
    }

def compare_models(X_train, X_test, y_train, y_test):
    """Compare different models"""
    models = [
        (RandomForestClassifier(n_estimators=300, max_depth=30, 
                              min_samples_split=2, min_samples_leaf=1,
                              class_weight='balanced_subsample', 
                              random_state=42), "Random Forest"),
        
        (GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                  learning_rate=0.1, random_state=42), 
         "Gradient Boosting"),
        
        (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                      random_state=42), 
         "Neural Network")
    ]
    
    results = []
    for model, name in models:
        result = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
    
    return results

def plot_model_comparison(results):
    """Plot model comparison results"""
    models = [r['model_name'] for r in results]
    scores = [r['cv_scores_mean'] for r in results]
    errors = [r['cv_scores_std'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(models, scores, yerr=errors, capsize=5)
    plt.title('Model Comparison - F1 Scores')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./output/model_comparison/model_comparison.png')
    plt.close()

def save_best_model(results, scaler):
    """Save the best performing model"""
    best_model = max(results, key=lambda x: x['cv_scores_mean'])
    print(f"\nBest performing model: {best_model['model_name']}")
    print(f"F1 Score: {best_model['cv_scores_mean']:.3f}")
    
    # Save model and scaler
    joblib.dump(best_model['model'], './models/best_model.pkl')
    joblib.dump(scaler, './models/best_scaler.pkl')
    print("\nBest model and scaler saved in 'models' directory")

def main():
    # Create output directory
    create_output_dir()
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(X, y)
    
    # Compare models
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Plot comparison
    plot_model_comparison(results)
    
    # Save best model
    save_best_model(results, scaler)
    
    print("\nModel comparison complete!")

if __name__ == "__main__":
    main() 