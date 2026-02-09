"""
Data Exploration and Understanding
===================================
This script analyzes the color dataset to understand its structure,
distribution, and quality before training.

💡 INTERVIEW TALKING POINTS:
- This is a multi-class classification problem
- Features: R, G, B (numerical, range 0-255)
- Target: ColorName (categorical)
- Understanding data is crucial before model selection
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import load_from_csv, load_color_dataset


def explore_dataset(df):
    """
    Comprehensive data exploration and analysis.
    
    Args:
        df (pd.DataFrame): Color dataset to explore
    """
    print("\n" + "="*70)
    print("📊 DATA EXPLORATION & UNDERSTANDING")
    print("="*70)
    
    # Basic Information
    print("\n1️⃣ BASIC INFORMATION")
    print("-" * 70)
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Records: {df.shape[0]:,}")
    print(f"Features: {df.shape[1]}")
    print(f"\nColumn Names: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    
    # Missing Values
    print("\n2️⃣ DATA QUALITY CHECK")
    print("-" * 70)
    missing = df.isnull().sum()
    print(f"Missing Values:\n{missing}")
    
    if missing.sum() == 0:
        print("✅ No missing values found - dataset is clean!")
    else:
        print("⚠️ Missing values detected - preprocessing required")
    
    # Statistical Summary
    print("\n3️⃣ NUMERICAL FEATURES STATISTICS (RGB Values)")
    print("-" * 70)
    if all(col in df.columns for col in ['R', 'G', 'B']):
        print(df[['R', 'G', 'B']].describe())
        
        print("\n📌 Expected RGB Range: 0-255")
        print(f"   R range: {df['R'].min()} - {df['R'].max()}")
        print(f"   G range: {df['G'].min()} - {df['G'].max()}")
        print(f"   B range: {df['B'].min()} - {df['B'].max()}")
    
    # Target Variable Analysis
    print("\n4️⃣ TARGET VARIABLE (ColorName) DISTRIBUTION")
    print("-" * 70)
    
    color_col = None
    for col in df.columns:
        if 'color' in col.lower() and 'name' in col.lower():
            color_col = col
            break
    
    if color_col is None:
        # Try other possible column names
        possible_names = ['ColorName', 'Color', 'Label', 'Class']
        for name in possible_names:
            if name in df.columns:
                color_col = name
                break
    
    if color_col:
        color_counts = df[color_col].value_counts()
        print(f"\nTotal Unique Colors: {df[color_col].nunique()}")
        print(f"\nTop 10 Most Frequent Colors:")
        print(color_counts.head(10))
        
        # Check for class imbalance
        print("\n5️⃣ CLASS BALANCE ANALYSIS")
        print("-" * 70)
        max_count = color_counts.max()
        min_count = color_counts.min()
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Most frequent color: {color_counts.index[0]} ({max_count} samples)")
        print(f"Least frequent color: {color_counts.index[-1]} ({min_count} samples)")
        print(f"Imbalance ratio: {ratio:.2f}:1")
        
        if ratio > 10:
            print("⚠️ Significant class imbalance detected")
            print("   Consider: SMOTE, class weights, or stratified sampling")
        else:
            print("✅ Classes are reasonably balanced")
    
    # Sample Data
    print("\n6️⃣ SAMPLE RECORDS")
    print("-" * 70)
    print(df.head(10))
    
    # Problem Definition
    print("\n" + "="*70)
    print("🎯 MACHINE LEARNING PROBLEM DEFINITION")
    print("="*70)
    print("""
📌 Problem Type: Multi-Class Classification

📌 Input Features (X):
   - R (Red channel): 0-255
   - G (Green channel): 0-255
   - B (Blue channel): 0-255
   
📌 Target Variable (y):
   - ColorName: Categorical (Red, Blue, Green, Yellow, etc.)
   
📌 Goal:
   Given RGB values, predict the correct color name.
   
📌 Why This Works:
   - Each color has unique RGB signature
   - RGB values form clusters in 3D space
   - ML models can learn decision boundaries
   
📌 Model Selection Reasoning:
   - Random Forest: Handles non-linear RGB boundaries
   - KNN: Works well for spatial clustering
   - SVM: Good for high-dimensional separation
    """)
    
    return df, color_col


def visualize_rgb_distribution(df):
    """
    Create visualizations of RGB distributions.
    
    Args:
        df (pd.DataFrame): Color dataset
    """
    if all(col in df.columns for col in ['R', 'G', 'B']):
        print("\n7️⃣ CREATING RGB DISTRIBUTION PLOTS...")
        print("-" * 70)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(df['R'], bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0].set_title('Red Channel Distribution')
        axes[0].set_xlabel('R Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(df['G'], bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('Green Channel Distribution')
        axes[1].set_xlabel('G Value')
        axes[1].set_ylabel('Frequency')
        
        axes[2].hist(df['B'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[2].set_title('Blue Channel Distribution')
        axes[2].set_xlabel('B Value')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('rgb_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ RGB distribution plot saved as 'rgb_distribution.png'")
        plt.close()


if __name__ == "__main__":
    # Load dataset
    df = load_from_csv()
    
    if df is None:
        print("Downloading dataset...")
        df = load_color_dataset()
        
        if df is not None:
            from load_dataset import save_dataset
            save_dataset(df)
    
    if df is not None:
        # Explore the dataset
        df, color_col = explore_dataset(df)
        
        # Create visualizations
        try:
            visualize_rgb_distribution(df)
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
        
        print("\n" + "="*70)
        print("✅ DATA EXPLORATION COMPLETE!")
        print("="*70)
        print("\nNext Steps:")
        print("1. Run preprocess.py to prepare data for training")
        print("2. Run train_model.py to train the classifier")
        print("3. Run predict_color.py to make predictions")
    else:
        print("❌ Could not load dataset. Please check load_dataset.py")
