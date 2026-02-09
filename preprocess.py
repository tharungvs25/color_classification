"""
Data Preprocessing for Color Classification
============================================
This module handles:
1. Feature selection (RGB values)
2. Label encoding (ColorName to numerical)
3. Train-test split
4. Data saving for model training

💡 INTERVIEW TALKING POINT:
"ML models require numerical inputs, so we encode text labels (ColorName)
 using LabelEncoder, which maintains consistency during prediction."
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from load_dataset import load_from_csv, load_color_dataset


class ColorDataPreprocessor:
    """
    Handles all preprocessing steps for color classification.
    """
    
    def __init__(self):
        self.encoder = LabelEncoder()
        self.feature_columns = ['R', 'G', 'B']
        self.target_column = None
        
    def find_target_column(self, df):
        """
        Automatically find the color name column.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            str: Name of the target column
        """
        # Try common column names
        possible_names = ['ColorName', 'Color Name', 'Color', 'Label', 'Class', 'color_name']
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # Find column with 'color' in name
        for col in df.columns:
            if 'color' in col.lower():
                return col
        
        # If still not found, return the last column (common convention)
        return df.columns[-1]
    
    def prepare_features_labels(self, df):
        """
        Extract features (X) and labels (y) from dataframe.
        
        Args:
            df (pd.DataFrame): Color dataset
            
        Returns:
            tuple: (X, y, target_column_name)
        """
        print("\n" + "="*70)
        print("🔧 PREPROCESSING DATA")
        print("="*70)
        
        # Find target column
        self.target_column = self.find_target_column(df)
        print(f"\n1️⃣ Target column identified: '{self.target_column}'")
        
        # Verify feature columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        
        # Extract features and labels
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        print(f"✅ Features (X) shape: {X.shape}")
        print(f"✅ Labels (y) shape: {y.shape}")
        print(f"\n   Feature columns: {self.feature_columns}")
        print(f"   Target column: {self.target_column}")
        
        return X, y, self.target_column
    
    def encode_labels(self, y):
        """
        Encode text labels to numerical values.
        
        Args:
            y (pd.Series): Color names
            
        Returns:
            np.array: Encoded labels
        """
        print(f"\n2️⃣ LABEL ENCODING")
        print("-" * 70)
        
        y_encoded = self.encoder.fit_transform(y)
        
        print(f"Total unique colors: {len(self.encoder.classes_)}")
        print(f"\nColor mappings (first 10):")
        for idx, color in enumerate(self.encoder.classes_[:10]):
            print(f"   {idx}: {color}")
        
        if len(self.encoder.classes_) > 10:
            print(f"   ... and {len(self.encoder.classes_) - 10} more colors")
        
        print(f"\n✅ Labels encoded successfully")
        print(f"   Encoded range: {y_encoded.min()} to {y_encoded.max()}")
        
        return y_encoded
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Labels (encoded)
            test_size (float): Proportion for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\n3️⃣ TRAIN-TEST SPLIT")
        print("-" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"Test size: {test_size * 100}%")
        print(f"Random state: {random_state} (for reproducibility)")
        print(f"\n✅ Data split successfully:")
        print(f"   Training samples: {X_train.shape[0]:,} ({(1-test_size)*100:.0f}%)")
        print(f"   Testing samples:  {X_test.shape[0]:,} ({test_size*100:.0f}%)")
        
        print(f"\n💡 Why this split?")
        print(f"   - Training data: Used to teach the model")
        print(f"   - Testing data: Used to evaluate generalization")
        print(f"   - Stratified: Maintains color distribution in both sets")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, 
                               output_dir='preprocessed_data'):
        """
        Save preprocessed data and encoder for later use.
        
        Args:
            X_train, X_test, y_train, y_test: Split datasets
            output_dir (str): Directory to save files
        """
        print(f"\n4️⃣ SAVING PREPROCESSED DATA")
        print("-" * 70)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save encoder (CRITICAL for prediction)
        joblib.dump(self.encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        
        print(f"✅ Data saved to '{output_dir}/' directory:")
        print(f"   - X_train.npy ({X_train.shape})")
        print(f"   - X_test.npy ({X_test.shape})")
        print(f"   - y_train.npy ({y_train.shape})")
        print(f"   - y_test.npy ({y_test.shape})")
        print(f"   - label_encoder.pkl (for predictions)")
        
    def load_preprocessed_data(self, input_dir='preprocessed_data'):
        """
        Load previously saved preprocessed data.
        
        Args:
            input_dir (str): Directory containing saved files
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, encoder)
        """
        try:
            X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
            encoder = joblib.load(os.path.join(input_dir, 'label_encoder.pkl'))
            
            print(f"✅ Loaded preprocessed data from '{input_dir}/'")
            return X_train, X_test, y_train, y_test, encoder
            
        except FileNotFoundError as e:
            print(f"❌ Error loading preprocessed data: {e}")
            print(f"   Run preprocessing first!")
            return None, None, None, None, None


def preprocess_pipeline(df=None, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Dataset (if None, will load from file)
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: Preprocessed data and encoder
    """
    # Load dataset if not provided
    if df is None:
        df = load_from_csv()
        if df is None:
            print("Downloading dataset...")
            df = load_color_dataset()
            if df is not None:
                from load_dataset import save_dataset
                save_dataset(df)
    
    if df is None:
        print("❌ Could not load dataset")
        return None
    
    # Create preprocessor
    preprocessor = ColorDataPreprocessor()
    
    # Execute preprocessing steps
    X, y, target_col = preprocessor.prepare_features_labels(df)
    y_encoded = preprocessor.encode_labels(y)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y_encoded, test_size, random_state
    )
    
    # Save everything
    preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nNext step: Run train_model.py to train the classifier")
    
    return X_train, X_test, y_train, y_test, preprocessor.encoder


if __name__ == "__main__":
    preprocess_pipeline()
