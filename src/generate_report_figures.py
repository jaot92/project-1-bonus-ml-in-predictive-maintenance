import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import joblib
import os

# Create output directories if they don't exist
os.makedirs('output/model_optimization', exist_ok=True)
os.makedirs('output/deployment', exist_ok=True)

def generate_cv_results():
    """Generate cross-validation results visualization"""
    # Load the model
    model = joblib.load('models/best_model.pkl')
    data = pd.read_csv('data/ai4i2020.csv')
    
    # Select only numeric features
    numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Prepare features
    X = data[numeric_features]
    y = data['Machine failure']
    
    # Perform cross-validation for each metric
    metrics = {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'f1'
    }
    
    cv_scores = {}
    for metric_name, scoring in metrics.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        cv_scores[metric_name] = scores
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame(cv_scores))
    plt.title('Cross-Validation Results Across 5 Folds')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('output/model_optimization/cv_results.png', bbox_inches='tight', dpi=300)
    plt.close()

def capture_web_interface():
    """Save a screenshot of the web interface"""
    # Since we can't take an actual screenshot, we'll create a mock visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a mock layout
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    # Add rectangles for different sections
    ax.add_patch(plt.Rectangle((0.5, 4), 5, 3.5, facecolor='#f8f9fa', edgecolor='black'))
    ax.add_patch(plt.Rectangle((6, 4), 5.5, 3.5, facecolor='#f8f9fa', edgecolor='black'))
    ax.add_patch(plt.Rectangle((0.5, 0.5), 5, 3, facecolor='#f8f9fa', edgecolor='black'))
    ax.add_patch(plt.Rectangle((6, 0.5), 5.5, 3, facecolor='#f8f9fa', edgecolor='black'))
    
    # Add text
    plt.text(3, 7, 'Input Parameters', ha='center', va='center', fontsize=12)
    plt.text(8.75, 7, 'Prediction Results', ha='center', va='center', fontsize=12)
    plt.text(3, 2, 'System Information', ha='center', va='center', fontsize=12)
    plt.text(8.75, 2, 'Parameter Health', ha='center', va='center', fontsize=12)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    plt.title('Predictive Maintenance Web Interface', pad=20)
    
    # Save plot
    plt.savefig('output/deployment/web_interface.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("Generating visualizations for the technical report...")
    
    # Generate cross-validation results
    print("Generating cross-validation results...")
    generate_cv_results()
    
    # Create web interface mockup
    print("Creating web interface visualization...")
    capture_web_interface()
    
    print("All visualizations have been generated successfully!")

if __name__ == '__main__':
    main() 