import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """
        Prepare data for training/prediction
        """
        # Add feature engineering steps here
        return df
    
    def train(self, X_train, y_train):
        """
        Train the model
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        """
        predictions = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
    def save_model(self, path):
        """
        Save the model to disk
        """
        joblib.dump(self.model, path)
        
    def load_model(self, path):
        """
        Load the model from disk
        """
        self.model = joblib.load(path)

if __name__ == "__main__":
    # Example usage
    # Load data
    df = pd.read_csv('../data/ai4i2020.csv')
    
    # Prepare features and target
    X = df.drop(['Machine failure'], axis=1)  # Adjust column names as needed
    y = df['Machine failure']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = PredictiveMaintenanceModel()
    model.train(X_train, y_train)
    
    # Evaluate
    model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model('../models/predictive_maintenance_model.pkl') 