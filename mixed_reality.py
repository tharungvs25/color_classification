"""
Mixed Reality Color Detection Module
=====================================
Real-time AR-style color detection using webcam feed.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from predict_color import ColorPredictor
import time


class MixedRealityColorDetector:
    """
    Handles real-time color detection with AR overlay.
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.detection_mode = "center"  # center, grid, or click
        self.show_boundaries = True
        self.show_fps = True
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def process_frame(self, frame, detection_points=None, detect_boundaries=False, target_colors=None):
        """
        Process a single frame with AR overlays.
        
        Args:
            frame: OpenCV BGR frame
            detection_points: List of (x, y) points to detect colors at
            detect_boundaries: Whether to detect color region boundaries
            target_colors: List of colors to detect boundaries for
            
        Returns:
            Processed frame with AR overlays
        """
        if frame is None:
            return None
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        # Create copy for drawing
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Add AR-style grid overlay
        self._draw_ar_grid(output_frame)
        
        # Center point detection (default)
        if detection_points is None or len(detection_points) == 0:
            detection_points = [(width // 2, height // 2)]
        
        # Detect and overlay colors at points
        for i, (x, y) in enumerate(detection_points):
            self._detect_and_overlay_color(output_frame, x, y, i + 1)
        
        # Detect color boundaries if enabled
        if detect_boundaries and target_colors:
            self._detect_and_draw_boundaries(output_frame, target_colors)
        
        # Add HUD (Heads-Up Display)
        self._draw_hud(output_frame)
        
        return output_frame
    
    def _draw_ar_grid(self, frame):
        """Draw AR-style grid overlay."""
        height, width = frame.shape[:2]
        
        # Draw subtle grid lines
        grid_color = (0, 255, 0)
        alpha = 0.3
        
        # Vertical lines
        for x in range(0, width, width // 4):
            cv2.line(frame, (x, 0), (x, height), grid_color, 1)
        
        # Horizontal lines
        for y in range(0, height, height // 4):
            cv2.line(frame, (0, y), (width, y), grid_color, 1)
        
        # Center crosshair
        center_x, center_y = width // 2, height // 2
        crosshair_size = 30
        cv2.line(frame, (center_x - crosshair_size, center_y), 
                (center_x + crosshair_size, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - crosshair_size), 
                (center_x, center_y + crosshair_size), (0, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), 2)
    
    def _detect_and_overlay_color(self, frame, x, y, index=1):
        """Detect color at point and draw AR overlay."""
        height, width = frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(5, min(x, width - 5))
        y = max(5, min(y, height - 5))
        
        # Sample color from region (5x5 average)
        region = frame[max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)]
        b, g, r = cv2.mean(region)[:3]
        
        # Predict color
        color_name, confidence = self.predictor.predict_with_confidence(int(r), int(g), int(b))
        
        # Draw detection point
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
        
        # Create AR-style label box
        label_text = f"{color_name.upper()}"
        rgb_text = f"RGB({int(r)},{int(g)},{int(b)})"
        conf_text = f"{confidence*100:.0f}%"
        
        # Calculate label position (above point)
        label_y = max(y - 100, 80)
        label_x = x - 100
        
        # Ensure label stays on screen
        if label_x < 10:
            label_x = 10
        if label_x > width - 220:
            label_x = width - 220
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x, label_y - 70), (label_x + 200, label_y + 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (label_x, label_y - 70), (label_x + 200, label_y + 10), 
                     (0, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, label_text, (label_x + 10, label_y - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, rgb_text, (label_x + 10, label_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Confidence: {conf_text}", (label_x + 10, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw connecting line
        cv2.line(frame, (x, y), (label_x + 100, label_y + 10), (0, 255, 255), 2)
    
    def _detect_and_draw_boundaries(self, frame, target_colors):
        """Detect and draw boundaries for specific colors."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color ranges
        color_ranges = {
            'Red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                   (np.array([160, 100, 100]), np.array([180, 255, 255]))],
            'Green': [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
            'Blue': [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
            'Yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        }
        
        for color_name in target_colors:
            if color_name not in color_ranges:
                continue
            
            # Create mask
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in color_ranges[color_name]:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            boundary_colors = {
                'Red': (0, 0, 255),
                'Green': (0, 255, 0),
                'Blue': (255, 0, 0),
                'Yellow': (0, 255, 255),
            }
            
            color = boundary_colors.get(color_name, (0, 255, 0))
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    # Draw contour
                    cv2.drawContours(frame, [contour], -1, color, 3)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Draw label
                    cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
                    cv2.putText(frame, color_name, (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_hud(self, frame):
        """Draw Heads-Up Display with info."""
        height, width = frame.shape[:2]
        
        # Top-left: Title
        cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (400, 60), (0, 255, 0), 2)
        cv2.putText(frame, "COLOR DETECTION AR", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "MIXED REALITY MODE", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Top-right: FPS
        fps_text = f"FPS: {self.fps}"
        cv2.rectangle(frame, (width - 150, 0), (width, 40), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 150, 0), (width, 40), (0, 255, 0), 2)
        cv2.putText(frame, fps_text, (width - 140, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Bottom: Instructions
        instructions = "Point at objects to detect colors | AI-Powered Detection"
        inst_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        inst_x = (width - inst_size[0]) // 2
        
        cv2.rectangle(frame, (inst_x - 10, height - 35), (inst_x + inst_size[0] + 10, height - 5), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (inst_x - 10, height - 35), (inst_x + inst_size[0] + 10, height - 5), 
                     (0, 255, 0), 2)
        cv2.putText(frame, instructions, (inst_x, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def create_grid_points(width, height, grid_size=3):
    """Create a grid of detection points."""
    points = []
    step_x = width // (grid_size + 1)
    step_y = height // (grid_size + 1)
    
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            points.append((step_x * i, step_y * j))
    
    return points
