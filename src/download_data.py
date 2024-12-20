import kagglehub
import os
import shutil

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    # Copy the dataset to our data directory
    for file in os.listdir(path):
        if file.endswith('.csv'):
            shutil.copy2(os.path.join(path, file), '../data/ai4i2020.csv')
            print(f"Dataset copied to: ../data/ai4i2020.csv")
            break
    
    print("Download completed!")

if __name__ == "__main__":
    download_dataset() 