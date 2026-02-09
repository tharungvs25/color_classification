"""
Create Sample Color Dataset
============================
Creates a synthetic color dataset for demonstration purposes.
This is a fallback when Kaggle dataset is unavailable.
"""

import pandas as pd
import numpy as np

def generate_color_dataset(num_samples=5000):
    """
    Generate synthetic color dataset with known RGB patterns.
    
    Args:
        num_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Dataset with R, G, B, and ColorName columns
    """
    print(f"Generating {num_samples} synthetic color samples...")
    
    data = []
    
    # Define color prototypes (R, G, B, Name)
    color_definitions = [
        # Primary colors
        ((255, 0, 0), 'Red', 30),
        ((0, 255, 0), 'Green', 30),
        ((0, 0, 255), 'Blue', 30),
        
        # Secondary colors
        ((255, 255, 0), 'Yellow', 25),
        ((255, 0, 255), 'Magenta', 25),
        ((0, 255, 255), 'Cyan', 25),
        
        # Tertiary and other colors
        ((255, 165, 0), 'Orange', 20),
        ((128, 0, 128), 'Purple', 20),
        ((255, 192, 203), 'Pink', 20),
        ((165, 42, 42), 'Brown', 15),
        ((128, 128, 0), 'Olive', 15),
        ((0, 128, 128), 'Teal', 15),
        
        # Neutrals
        ((255, 255, 255), 'White', 30),
        ((0, 0, 0), 'Black', 30),
        ((128, 128, 128), 'Gray', 25),
        
        # Shades
        ((255, 20, 147), 'Deep Pink', 15),
        ((255, 69, 0), 'Red Orange', 15),
        ((50, 205, 50), 'Lime Green', 15),
        ((0, 0, 128), 'Navy Blue', 15),
        ((148, 0, 211), 'Dark Violet', 15),
        ((255, 215, 0), 'Gold', 15),
        ((192, 192, 192), 'Silver', 15),
        ((139, 69, 19), 'Saddle Brown', 10),
        ((255, 99, 71), 'Tomato', 10),
        ((34, 139, 34), 'Forest Green', 10),
    ]
    
    for (r_base, g_base, b_base), color_name, num_variants in color_definitions:
        for _ in range(num_variants):
            # Add some variation to make dataset realistic
            noise_level = 30
            r = np.clip(r_base + np.random.randint(-noise_level, noise_level), 0, 255)
            g = np.clip(g_base + np.random.randint(-noise_level, noise_level), 0, 255)
            b = np.clip(b_base + np.random.randint(-noise_level, noise_level), 0, 255)
            
            data.append({
                'R': int(r),
                'G': int(g),
                'B': int(b),
                'ColorName': color_name
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} samples across {df['ColorName'].nunique()} colors")
    print(f"\nColor distribution:")
    print(df['ColorName'].value_counts().head(10))
    
    return df


if __name__ == "__main__":
    # Generate dataset
    df = generate_color_dataset(num_samples=5000)
    
    # Save to CSV
    filename = 'color_dataset.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 Dataset saved to: {filename}")
    
    # Display sample
    print(f"\nSample data:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
