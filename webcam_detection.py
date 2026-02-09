"""
Real-Time Color Detection using Webcam
=======================================
This script captures video from webcam and detects colors in real-time.
The center pixel color is analyzed and displayed on screen.

💡 INTERVIEW TALKING POINTS:
- Real-time application of ML model
- OpenCV for video capture and processing
- Demonstrates practical use case beyond static predictions
- Can be extended to object detection, AR applications, etc.

CONTROLS:
- Press 'q' to quit
- Press 'c' to capture current frame
- Press 's' to save screenshot
"""

import cv2
import numpy as np
from predict_color import ColorPredictor
import os
from datetime import datetime


class WebcamColorDetector:
    """
    Real-time color detection from webcam feed.
    """
    
    def __init__(self, model_path='models/random_forest_color_classifier.pkl',
                 encoder_path='preprocessed_data/label_encoder.pkl'):
        """
        Initialize webcam detector.
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to label encoder
        """
        self.predictor = ColorPredictor(model_path, encoder_path)
        self.cap = None
        self.is_running = False
        
        # Display settings
        self.window_name = "Color Detection - Press 'q' to quit"
        self.target_fps = 30
        self.frame_delay = int(1000 / self.target_fps)
        
        # Detection settings
        self.detection_radius = 30  # Radius of detection circle
        self.sample_size = 5  # Average color from 5x5 pixel area
        
    def initialize_camera(self, camera_id=0):
        """
        Initialize webcam capture.
        
        Args:
            camera_id (int): Camera device ID (0 for default)
            
        Returns:
            bool: True if successful
        """
        print(f"\n📹 Initializing camera {camera_id}...")
        
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
            print("   Possible solutions:")
            print("   1. Check if camera is connected")
            print("   2. Close other applications using camera")
            print("   3. Try different camera_id (1, 2, etc.)")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
        return True
    
    def get_center_color(self, frame):
        """
        Extract color from center region of frame.
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            tuple: (r, g, b) values
        """
        h, w, _ = frame.shape
        center_y, center_x = h // 2, w // 2
        
        # Sample from a small region (reduces noise)
        half_sample = self.sample_size // 2
        y1 = max(0, center_y - half_sample)
        y2 = min(h, center_y + half_sample)
        x1 = max(0, center_x - half_sample)
        x2 = min(w, center_x + half_sample)
        
        # Extract region and calculate mean color
        region = frame[y1:y2, x1:x2]
        
        # OpenCV uses BGR, convert to RGB
        b, g, r = cv2.mean(region)[:3]
        
        return int(r), int(g), int(b)
    
    def draw_ui(self, frame, color_name, r, g, b, confidence=None):
        """
        Draw UI elements on frame.
        
        Args:
            frame: Video frame
            color_name: Predicted color name
            r, g, b: RGB values
            confidence: Prediction confidence (optional)
        """
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        # Draw detection circle
        cv2.circle(frame, (center_x, center_y), self.detection_radius, 
                   (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Top info panel
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Color sample box
        box_size = 80
        box_x = w - box_size - 20
        box_y = 20
        cv2.rectangle(overlay, (box_x, box_y), 
                     (box_x + box_size, box_y + box_size), 
                     (int(b), int(g), int(r)), -1)
        cv2.rectangle(frame, (box_x, box_y), 
                     (box_x + box_size, box_y + box_size), 
                     (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Color name (large)
        cv2.putText(frame, color_name.upper(), (20, 50),
                   font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # RGB values
        rgb_text = f"RGB: ({r}, {g}, {b})"
        cv2.putText(frame, rgb_text, (20, 85),
                   font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Confidence (if available)
        if confidence is not None and confidence < 1.0:
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (20, 110),
                       font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Bottom instructions
        instructions = "Press 'q' to quit | 's' to save | 'c' to capture"
        cv2.putText(frame, instructions, (20, h - 20),
                   font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def save_frame(self, frame, color_name):
        """
        Save current frame as image.
        
        Args:
            frame: Video frame to save
            color_name: Detected color name
        """
        # Create screenshots directory
        os.makedirs('screenshots', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/{color_name}_{timestamp}.png"
        
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def run(self):
        """
        Main loop for real-time color detection.
        """
        print("\n" + "="*70)
        print("🎨 REAL-TIME COLOR DETECTION")
        print("="*70)
        
        # Load model
        if not self.predictor.load_model():
            return
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        print("\n✅ Starting color detection...")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'c' to capture and print color info")
        print("\n" + "-"*70)
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Cannot receive frame from camera")
                    break
                
                # Get color from center
                r, g, b = self.get_center_color(frame)
                
                # Predict color
                color_name, confidence = self.predictor.predict_with_confidence(r, g, b)
                
                # Draw UI
                frame = self.draw_ui(frame, color_name, r, g, b, confidence)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(self.frame_delay) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(frame, color_name)
                elif key == ord('c'):
                    print(f"\n📷 Captured: RGB({r}, {g}, {b}) → {color_name.upper()}")
                    if confidence < 1.0:
                        print(f"   Confidence: {confidence*100:.2f}%")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        
        finally:
            # Cleanup
            self.cleanup()
            print(f"\nProcessed {frame_count} frames")
            print("="*70)
    
    def cleanup(self):
        """
        Release resources.
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False


def test_static_image(image_path):
    """
    Test color detection on a static image.
    
    Args:
        image_path (str): Path to image file
    """
    print(f"\n🖼️  Testing on image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Cannot load image")
        return
    
    # Initialize detector
    detector = WebcamColorDetector()
    if not detector.predictor.load_model():
        return
    
    # Get center color
    r, g, b = detector.get_center_color(frame)
    color_name, confidence = detector.predictor.predict_with_confidence(r, g, b)
    
    # Draw UI
    frame = detector.draw_ui(frame, color_name, r, g, b, confidence)
    
    # Display
    print(f"✅ Detected: RGB({r}, {g}, {b}) → {color_name.upper()}")
    if confidence < 1.0:
        print(f"   Confidence: {confidence*100:.2f}%")
    
    cv2.imshow("Color Detection - Press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on image
        test_static_image(sys.argv[1])
    else:
        # Run webcam detection
        detector = WebcamColorDetector()
        detector.run()
