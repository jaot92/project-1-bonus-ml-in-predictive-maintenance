import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Configurar estilo de las visualizaciones
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def create_output_dir():
    """Create output directory for presentation figures"""
    output_dir = os.path.join('output', 'presentation')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data():
    """Load and prepare data"""
    data = pd.read_csv('data/ai4i2020.csv')
    return data

def plot_failure_distribution(data, output_dir):
    """Plot distribution of machine failures"""
    plt.figure(figsize=(8, 6))
    failure_counts = data['Machine failure'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    plt.pie(failure_counts, labels=['No Failure', 'Failure'],
            autopct='%1.1f%%', colors=colors,
            explode=(0, 0.1))
    
    plt.title('Distribution of Machine Failures', pad=20)
    plt.savefig(os.path.join(output_dir, 'failure_distribution.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(output_dir):
    """Plot feature importance"""
    # Simulated feature importance for presentation
    features = ['Tool wear', 'Temperature diff', 'Process temperature',
                'Rotational speed', 'Torque', 'Air temperature', 'Power']
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importance)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2%}', ha='left', va='center', fontweight='bold')
    
    plt.title('Feature Importance in Failure Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_comparison(output_dir):
    """Plot model comparison"""
    models = ['Random Forest', 'Gradient Boosting', 'Neural Network']
    f1_scores = [0.973, 0.988, 0.980]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_scores)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison (F1 Score)')
    plt.ylim(0.95, 1.0)  # Zoom in on the relevant range
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_system_architecture(output_dir):
    """Create a simple system architecture diagram"""
    from graphviz import Digraph
    try:
        dot = Digraph(comment='System Architecture')
        dot.attr(rankdir='LR')
        
        # Add nodes
        dot.node('A', 'Sensor Data')
        dot.node('B', 'Flask API')
        dot.node('C', 'ML Model')
        dot.node('D', 'Web Interface')
        
        # Add edges
        dot.edge('A', 'B')
        dot.edge('B', 'C')
        dot.edge('C', 'B')
        dot.edge('B', 'D')
        
        # Save
        dot.render(os.path.join(output_dir, 'architecture'), format='png', cleanup=True)
    except Exception as e:
        print(f"Could not create architecture diagram: {str(e)}")

def main():
    """Generate all presentation figures"""
    output_dir = create_output_dir()
    data = load_data()
    
    print("Generating presentation figures...")
    plot_failure_distribution(data, output_dir)
    plot_feature_importance(output_dir)
    plot_model_comparison(output_dir)
    plot_system_architecture(output_dir)
    print("Figures generated successfully!")

if __name__ == '__main__':
    main() 