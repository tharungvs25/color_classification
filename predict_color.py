"""
Color Prediction Module
========================
This module loads the trained model and makes predictions on new RGB values.
Supports both single predictions and batch predictions.

💡 INTERVIEW TALKING POINT:
"We use the saved LabelEncoder to convert model's numerical output
 back to human-readable color names, ensuring consistency with training."
"""

import numpy as np
import joblib
import os
from typing import Union, List, Tuple


class ColorPredictor:
    """
    Handles color predictions from RGB values using trained model.
    """
    
    def __init__(self, model_path='models/random_forest_color_classifier.pkl',
                 encoder_path='preprocessed_data/label_encoder.pkl'):
        """
        Initialize predictor with trained model and encoder.
        
        Args:
            model_path (str): Path to saved model
            encoder_path (str): Path to saved label encoder
        """
        self.model = None
        self.encoder = None
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.is_loaded = False
        
    def load_model(self):
        """
        Load trained model and encoder.
        
        Returns:
            bool: True if loading successful
        """
        try:
            # Load model
            if not os.path.exists(self.model_path):
                print(f"❌ Model not found: {self.model_path}")
                print("   Run train_model.py first!")
                return False
            
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            
            # Load encoder
            if not os.path.exists(self.encoder_path):
                print(f"❌ Encoder not found: {self.encoder_path}")
                print("   Run preprocess.py first!")
                return False
            
            self.encoder = joblib.load(self.encoder_path)
            print(f"✅ Encoder loaded: {self.encoder_path}")
            print(f"   Available colors: {len(self.encoder.classes_)}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_color(self, r: int, g: int, b: int) -> str:
        """
        Predict color name from RGB values.
        
        Args:
            r (int): Red value (0-255)
            g (int): Green value (0-255)
            b (int): Blue value (0-255)
            
        Returns:
            str: Predicted color name
        """
        if not self.is_loaded:
            if not self.load_model():
                return "Error: Model not loaded"
        
        # Validate input
        if not all(0 <= val <= 255 for val in [r, g, b]):
            return "Error: RGB values must be between 0-255"
        
        # Prepare input
        rgb_array = np.array([[r, g, b]])
        
        # Predict
        prediction_encoded = self.model.predict(rgb_array)
        color_name = self.encoder.inverse_transform(prediction_encoded)[0]
        
        return color_name
    
    def predict_batch(self, rgb_list: List[Tuple[int, int, int]]) -> List[str]:
        """
        Predict colors for multiple RGB values.
        
        Args:
            rgb_list: List of (R, G, B) tuples
            
        Returns:
            List of predicted color names
        """
        if not self.is_loaded:
            if not self.load_model():
                return ["Error: Model not loaded"] * len(rgb_list)
        
        # Convert to numpy array
        rgb_array = np.array(rgb_list)
        
        # Predict
        predictions_encoded = self.model.predict(rgb_array)
        color_names = self.encoder.inverse_transform(predictions_encoded)
        
        return color_names.tolist()
    
    def predict_with_confidence(self, r: int, g: int, b: int) -> Tuple[str, float]:
        """
        Predict color with confidence score (for probabilistic models).
        
        Args:
            r, g, b: RGB values
            
        Returns:
            tuple: (color_name, confidence)
        """
        if not self.is_loaded:
            if not self.load_model():
                return "Error: Model not loaded", 0.0
        
        # Validate input
        if not all(0 <= val <= 255 for val in [r, g, b]):
            return "Error: RGB values must be between 0-255", 0.0
        
        rgb_array = np.array([[r, g, b]])
        
        # Get prediction
        prediction_encoded = self.model.predict(rgb_array)
        color_name = self.encoder.inverse_transform(prediction_encoded)[0]
        
        # Get confidence (if model supports predict_proba)
        try:
            probabilities = self.model.predict_proba(rgb_array)
            confidence = np.max(probabilities)
        except AttributeError:
            confidence = 1.0  # If model doesn't support probabilities
        
        return color_name, confidence
    
    def get_available_colors(self) -> List[str]:
        """
        Get list of all colors the model can predict.
        
        Returns:
            List of color names
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        return self.encoder.classes_.tolist()


def predict_color_from_pixel(r: int, g: int, b: int) -> str:
    """
    Convenience function to predict color from RGB values.
    
    Args:
        r, g, b: RGB values (0-255)
        
    Returns:
        str: Predicted color name
    """
    predictor = ColorPredictor()
    return predictor.predict_color(r, g, b)


def interactive_prediction():
    """
    Interactive mode for testing color predictions.
    """
    print("\n" + "="*70)
    print("🎨 COLOR PREDICTION - INTERACTIVE MODE")
    print("="*70)
    
    predictor = ColorPredictor()
    if not predictor.load_model():
        return
    
    print(f"\n✅ Ready to predict!")
    print(f"Available colors: {len(predictor.get_available_colors())}")
    print("\nExamples to try:")
    print("  - Pure Red:    255, 0, 0")
    print("  - Pure Green:  0, 255, 0")
    print("  - Pure Blue:   0, 0, 255")
    print("  - White:       255, 255, 255")
    print("  - Black:       0, 0, 0")
    print("  - Yellow:      255, 255, 0")
    
    print("\n" + "-"*70)
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter RGB values (e.g., 255,0,0): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Parse input
            rgb_values = [int(x.strip()) for x in user_input.split(',')]
            
            if len(rgb_values) != 3:
                print("❌ Please enter exactly 3 values (R, G, B)")
                continue
            
            r, g, b = rgb_values
            
            # Predict with confidence
            color, confidence = predictor.predict_with_confidence(r, g, b)
            
            print(f"\n🎨 RGB({r}, {g}, {b}) → {color.upper()}")
            if confidence < 1.0:
                print(f"   Confidence: {confidence*100:.2f}%")
            print()
            
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_common_colors():
    """
    Test predictions on common colors.
    """
    print("\n" + "="*70)
    print("🧪 TESTING COMMON COLORS")
    print("="*70)
    
    predictor = ColorPredictor()
    if not predictor.load_model():
        return
    
    # Common colors to test
    test_colors = [
        (255, 0, 0, "Pure Red"),
        (0, 255, 0, "Pure Green"),
        (0, 0, 255, "Pure Blue"),
        (255, 255, 0, "Yellow"),
        (255, 0, 255, "Magenta"),
        (0, 255, 255, "Cyan"),
        (255, 255, 255, "White"),
        (0, 0, 0, "Black"),
        (128, 128, 128, "Gray"),
        (255, 165, 0, "Orange"),
        (128, 0, 128, "Purple"),
        (165, 42, 42, "Brown"),
        (255, 192, 203, "Pink"),
    ]
    
    print("\n" + "-"*70)
    print(f"{'Expected Color':<20} {'RGB':<20} {'Predicted':<20} {'Conf.':<10}")
    print("-"*70)
    
    for r, g, b, expected in test_colors:
        predicted, conf = predictor.predict_with_confidence(r, g, b)
        conf_str = f"{conf*100:.1f}%" if conf < 1.0 else "-"
        print(f"{expected:<20} RGB({r:3d},{g:3d},{b:3d})  →  {predicted:<20} {conf_str:<10}")
    
    print("-"*70)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Test mode
            test_common_colors()
        elif len(sys.argv) == 4:
            # Direct prediction: python predict_color.py 255 0 0
            try:
                r, g, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
                predictor = ColorPredictor()
                if predictor.load_model():
                    color, conf = predictor.predict_with_confidence(r, g, b)
                    print(f"\nRGB({r}, {g}, {b}) → {color.upper()}")
                    if conf < 1.0:
                        print(f"Confidence: {conf*100:.2f}%")
            except ValueError:
                print("Usage: python predict_color.py <R> <G> <B>")
                print("Example: python predict_color.py 255 0 0")
        else:
            print("Usage:")
            print("  python predict_color.py              # Interactive mode")
            print("  python predict_color.py test         # Test common colors")
            print("  python predict_color.py <R> <G> <B>  # Direct prediction")
    else:
        # Interactive mode
        interactive_prediction()
