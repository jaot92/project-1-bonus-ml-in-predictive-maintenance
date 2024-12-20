import pandas as pd
import os

def check_dataset():
    """
    Check if the dataset exists and can be read correctly
    """
    data_path = '../data/ai4i2020.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in the data folder.")
        return False
    
    try:
        # Try to read the dataset
        df = pd.read_csv(data_path)
        
        # Print basic information
        print("\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        return True
    
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    check_dataset() 