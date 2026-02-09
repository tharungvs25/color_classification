"""
Model Training for Color Classification
========================================
This script trains a Random Forest classifier to predict color names
from RGB values.

💡 INTERVIEW TALKING POINTS:
- Random Forest chosen for handling non-linear RGB decision boundaries
- Ensemble method: combines multiple decision trees
- Robust to overfitting and works well with small-medium datasets
- Can handle multi-class classification naturally
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import joblib
import os
import time
from preprocess import ColorDataPreprocessor


class ColorClassifier:
    """
    Wrapper class for training and evaluating color classification model.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize classifier.
        
        Args:
            model_type (str): Type of model ('random_forest', 'knn', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.training_time = 0
        self.metrics = {}
        
    def create_model(self, **kwargs):
        """
        Create the ML model based on type.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            Trained model object
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),  # Use all CPU cores
                verbose=kwargs.get('verbose', 0)
            )
            print("\n🌲 Random Forest Classifier Created")
            print(f"   - Trees: {kwargs.get('n_estimators', 100)}")
            print(f"   - Max Depth: {kwargs.get('max_depth', 'None (unlimited)')}")
            print(f"   - Random State: {kwargs.get('random_state', 42)}")
            
        elif self.model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                n_jobs=-1
            )
            print(f"\n🎯 K-Nearest Neighbors Classifier Created")
            print(f"   - Neighbors (k): {kwargs.get('n_neighbors', 5)}")
            
        elif self.model_type == 'svm':
            from sklearn.svm import SVC
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                random_state=kwargs.get('random_state', 42)
            )
            print(f"\n⚡ Support Vector Machine Created")
            print(f"   - Kernel: {kwargs.get('kernel', 'rbf')}")
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*70)
        print("🚀 TRAINING MODEL")
        print("="*70)
        
        if self.model is None:
            self.create_model()
        
        print(f"\nTraining on {X_train.shape[0]:,} samples...")
        print(f"Features: {X_train.shape[1]} (R, G, B)")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {self.training_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test, encoder):
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels (encoded)
            encoder: LabelEncoder to decode predictions
        """
        print("\n" + "="*70)
        print("📊 MODEL EVALUATION")
        print("="*70)
        
        # Make predictions
        print(f"\nPredicting on {X_test.shape[0]:,} test samples...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'test_samples': len(y_test),
            'training_time': self.training_time
        }
        
        print(f"\n✅ PERFORMANCE METRICS")
        print("-" * 70)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT")
        print("-" * 70)
        target_names = encoder.classes_
        print(classification_report(y_test, y_pred, 
                                   target_names=target_names,
                                   zero_division=0))
        
        # Feature importance (for Random Forest)
        if self.model_type == 'random_forest':
            print(f"\n🎯 FEATURE IMPORTANCE")
            print("-" * 70)
            importances = self.model.feature_importances_
            features = ['Red (R)', 'Green (G)', 'Blue (B)']
            for feat, imp in zip(features, importances):
                print(f"{feat:12s}: {imp:.4f} {'█' * int(imp * 50)}")
        
        return self.metrics
    
    def save_model(self, filepath='models/color_classifier.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"\n💾 Model saved to: {filepath}")
        
    def load_model(self, filepath='models/color_classifier.pkl'):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"✅ Model loaded from: {filepath}")
            return self.model
        else:
            print(f"❌ Model file not found: {filepath}")
            return None


def train_pipeline(model_type='random_forest', **model_kwargs):
    """
    Complete training pipeline.
    
    Args:
        model_type (str): Type of model to train
        **model_kwargs: Additional model parameters
    """
    print("\n" + "="*70)
    print("🎨 COLOR CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Load preprocessed data
    print("\n1️⃣ Loading preprocessed data...")
    preprocessor = ColorDataPreprocessor()
    X_train, X_test, y_train, y_test, encoder = preprocessor.load_preprocessed_data()
    
    if X_train is None:
        print("\n⚠️ Preprocessed data not found. Running preprocessing...")
        from preprocess import preprocess_pipeline
        X_train, X_test, y_train, y_test, encoder = preprocess_pipeline()
    
    if X_train is None:
        print("❌ Failed to load or preprocess data")
        return None, None
    
    # Create and train model
    print("\n2️⃣ Initializing model...")
    classifier = ColorClassifier(model_type=model_type)
    classifier.create_model(**model_kwargs)
    
    print("\n3️⃣ Training...")
    classifier.train(X_train, y_train)
    
    # Evaluate model
    print("\n4️⃣ Evaluating...")
    metrics = classifier.evaluate(X_test, y_test, encoder)
    
    # Save model
    print("\n5️⃣ Saving model...")
    model_path = f'models/{model_type}_color_classifier.pkl'
    classifier.save_model(model_path)
    
    # Also save encoder path reference
    print("\n💡 Remember: The label encoder is saved in 'preprocessed_data/label_encoder.pkl'")
    print("   You'll need it for making predictions!")
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nModel Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Training Time: {metrics['training_time']:.2f}s")
    print(f"\nNext steps:")
    print("1. Run predict_color.py to test predictions")
    print("2. Run webcam_detection.py for real-time detection")
    
    return classifier, encoder


def compare_models():
    """
    Train and compare multiple models.
    """
    print("\n" + "="*70)
    print("🔬 MODEL COMPARISON")
    print("="*70)
    
    models_to_test = [
        ('random_forest', {'n_estimators': 100}),
        ('knn', {'n_neighbors': 5}),
    ]
    
    results = []
    
    for model_type, params in models_to_test:
        print(f"\n\n{'='*70}")
        print(f"Testing: {model_type.upper()}")
        print(f"{'='*70}")
        
        classifier, encoder = train_pipeline(model_type, **params)
        
        if classifier:
            results.append({
                'model': model_type,
                'accuracy': classifier.metrics['accuracy'],
                'f1_score': classifier.metrics['f1_score'],
                'training_time': classifier.metrics['training_time']
            })
    
    # Print comparison
    print("\n\n" + "="*70)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    best_model = df_results.loc[df_results['accuracy'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['model'].upper()}")
    print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")


if __name__ == "__main__":
    # Train default model (Random Forest)
    train_pipeline(
        model_type='random_forest',
        n_estimators=100,
        random_state=42
    )
    
    # Uncomment to compare multiple models:
    # compare_models()
