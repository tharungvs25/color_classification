"""
Dataset Loader for Color Classification Project
================================================
This script loads the Color Dataset from Kaggle using mlcroissant
and converts it to a pandas DataFrame for further processing.

Dataset: Color Dataset for Color Recognition
Link: https://www.kaggle.com/datasets/adikurniawan/color-dataset-for-color-recognition
"""

import mlcroissant as mlc
import pandas as pd
import os


def load_color_dataset():
    """
    Load the color dataset using Croissant metadata format.
    
    Returns:
        pd.DataFrame: DataFrame containing RGB values and color names
    """
    print("Loading Color Dataset from Kaggle...")
    
    try:
        # Load Croissant metadata
        dataset = mlc.Dataset(
            "https://www.kaggle.com/datasets/adikurniawan/color-dataset-for-color-recognition/croissant/download"
        )

        # Inspect record sets
        record_sets = dataset.metadata.record_sets
        print(f"Available Record Sets: {len(record_sets)}")
        
        if record_sets:
            print(f"Using record set: {record_sets[0].uuid}")

        # Convert first record set to DataFrame
        df = pd.DataFrame(dataset.records(record_set=record_sets[0].uuid))

        print("\n✅ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTip: Make sure you have:")
        print("1. Installed mlcroissant: pip install mlcroissant")
        print("2. Stable internet connection")
        print("3. Kaggle API configured (if required)")
        return None


def save_dataset(df, filename='color_dataset.csv'):
    """
    Save the loaded dataset to CSV for faster loading in future.
    
    Args:
        df (pd.DataFrame): Dataset to save
        filename (str): Output filename
    """
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\n💾 Dataset saved to {filename}")


def load_from_csv(filename='color_dataset.csv'):
    """
    Load dataset from saved CSV file.
    
    Args:
        filename (str): CSV file to load
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if os.path.exists(filename):
        print(f"Loading dataset from {filename}...")
        df = pd.read_csv(filename)
        print(f"✅ Loaded {df.shape[0]} records")
        return df
    else:
        print(f"❌ File {filename} not found")
        return None


if __name__ == "__main__":
    # Try loading from cached CSV first
    df = load_from_csv()
    
    # If not found, download from Kaggle
    if df is None:
        df = load_color_dataset()
        
        if df is not None:
            # Display basic info
            print("\n" + "="*60)
            print("DATASET PREVIEW")
            print("="*60)
            print(df.head())
            
            # Save for future use
            save_dataset(df)
