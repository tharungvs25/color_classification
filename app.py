"""
Color Classification Web UI
============================
Interactive Streamlit web application for color prediction.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_color import ColorPredictor
from sklearn.cluster import KMeans
from mixed_reality import MixedRealityColorDetector, create_grid_points
import cv2
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="Color Classification AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .color-box {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #666;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and encoder."""
    import sys
    from io import StringIO
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = output_capture = StringIO()
    
    try:
        predictor = ColorPredictor()
        success = predictor.load_model()
        
        # Restore stdout
        sys.stdout = old_stdout
        output = output_capture.getvalue()
        
        # Log output in debug mode
        if output:
            print("Model loading output:", output)
        
        if success and predictor.is_loaded:
            return predictor
        else:
            print(f"Model loading failed: success={success}, is_loaded={predictor.is_loaded if predictor else 'N/A'}")
            return None
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Exception during model loading: {e}")
        import traceback
        traceback.print_exc()
        return None

# Get predictor (cached)
predictor = load_model()

def extract_dominant_colors(image, num_colors=5):
    """
    Extract dominant colors from an image using K-means clustering.
    
    Args:
        image: PIL Image object
        num_colors: Number of dominant colors to extract
        
    Returns:
        list of tuples: [(r, g, b, percentage), ...]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for faster processing
    img = image.copy()
    img.thumbnail((300, 300))
    
    # Convert to numpy array and reshape
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    # Remove very dark pixels (likely background)
    mask = pixels.sum(axis=1) > 30
    pixels = pixels[mask]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Count pixels in each cluster to get percentages
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    
    # Sort by percentage (most dominant first)
    sorted_indices = np.argsort(-percentages)
    
    results = []
    for idx in sorted_indices:
        r, g, b = colors[idx]
        percentage = percentages[idx]
        results.append((int(r), int(g), int(b), float(percentage)))
    
    return results

def annotate_image_with_colors(image, dominant_colors, predictor):
    """
    Annotate image with color labels and markers.
    
    Args:
        image: PIL Image object
        dominant_colors: List of (r, g, b, percentage) tuples
        predictor: ColorPredictor instance
        
    Returns:
        PIL Image: Annotated image
    """
    # Create a copy to draw on
    img_annotated = image.copy()
    draw = ImageDraw.Draw(img_annotated)
    
    # Get image dimensions
    width, height = img_annotated.size
    
    # Try to load a font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 25)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Create color palette on the right side
    palette_width = 300
    color_height = height // len(dominant_colors)
    
    for i, (r, g, b, percentage) in enumerate(dominant_colors):
        # Predict color name
        color_name, confidence = predictor.predict_with_confidence(r, g, b)
        
        # Draw color box on the right side
        x1 = width - palette_width
        y1 = i * color_height
        x2 = width
        y2 = (i + 1) * color_height
        
        # Draw the color box
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))
        
        # Add border
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)
        
        # Determine text color (white or black) based on background brightness
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # Draw color name
        text_y = y1 + color_height // 2 - 30
        draw.text((x1 + 10, text_y), color_name.upper(), fill=text_color, font=font_large)
        
        # Draw RGB values
        rgb_text = f"RGB({r}, {g}, {b})"
        draw.text((x1 + 10, text_y + 45), rgb_text, fill=text_color, font=font_small)
        
        # Draw percentage
        pct_text = f"{percentage:.1f}% of image"
        draw.text((x1 + 10, text_y + 75), pct_text, fill=text_color, font=font_small)
    
    # Add title banner at top
    banner_height = 60
    draw.rectangle([0, 0, width, banner_height], fill=(0, 0, 0, 200))
    title_text = f"🎨 {len(dominant_colors)} Dominant Colors Detected"
    draw.text((20, 15), title_text, fill=(255, 255, 255), font=font_large)
    
    return img_annotated

def get_color_ranges():
    """
    Define HSV color ranges for different colors.
    Returns a dictionary of color names and their HSV ranges.
    """
    color_ranges = {
        'Red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))  # Red wraps around
        ],
        'Green': [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
        'Blue': [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
        'Yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        'Orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
        'Purple': [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
        'Pink': [(np.array([145, 50, 50]), np.array([170, 255, 255]))],
        'Cyan': [(np.array([85, 50, 50]), np.array([95, 255, 255]))],
        'White': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
        'Black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
        'Gray': [(np.array([0, 0, 50]), np.array([180, 30, 200]))],
        'Brown': [(np.array([10, 100, 20]), np.array([20, 255, 200]))],
    }
    return color_ranges

def detect_color_regions(image, target_colors=None, min_area=500):
    """
    Detect regions of specific colors in an image.
    
    Args:
        image: PIL Image object
        target_colors: List of color names to detect (None = detect all)
        min_area: Minimum contour area to consider
        
    Returns:
        List of detected regions with color, contour, and bounding box
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    color_ranges = get_color_ranges()
    
    # Filter by target colors if specified
    if target_colors:
        color_ranges = {k: v for k, v in color_ranges.items() if k in target_colors}
    
    detected_regions = []
    
    for color_name, ranges in color_ranges.items():
        # Create mask for this color
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in ranges:
            color_mask = cv2.inRange(img_hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                detected_regions.append({
                    'color': color_name,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area
                })
    
    return detected_regions

def draw_color_boundaries(image, detected_regions):
    """
    Draw boundaries and labels for detected color regions.
    
    Args:
        image: PIL Image object
        detected_regions: List of detected regions
        
    Returns:
        PIL Image with boundaries drawn
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Group regions by color for numbering
    color_counts = {}
    
    for region in detected_regions:
        color_name = region['color']
        x, y, w, h = region['bbox']
        contour = region['contour']
        
        # Count occurrences of each color
        if color_name not in color_counts:
            color_counts[color_name] = 0
        color_counts[color_name] += 1
        
        # Choose color for boundary (contrasting colors)
        boundary_colors = {
            'Red': (255, 0, 0),
            'Green': (0, 255, 0),
            'Blue': (0, 0, 255),
            'Yellow': (255, 255, 0),
            'Orange': (255, 165, 0),
            'Purple': (128, 0, 128),
            'Pink': (255, 192, 203),
            'Cyan': (0, 255, 255),
            'White': (255, 255, 255),
            'Black': (0, 0, 0),
            'Gray': (128, 128, 128),
            'Brown': (165, 42, 42),
        }
        
        boundary_color = boundary_colors.get(color_name, (0, 255, 0))
        
        # Draw contour
        cv2.drawContours(img_cv, [contour], -1, boundary_color, 3)
        
        # Draw bounding box
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), boundary_color, 2)
        
        # Prepare label
        label = f"{color_name} #{color_counts[color_name]}"
        
        # Calculate label position (above the bounding box)
        label_y = max(y - 10, 30)
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(img_cv, 
                     (x, label_y - text_size[1] - 10),
                     (x + text_size[0] + 10, label_y + 5),
                     boundary_color, -1)
        
        # Draw label text
        text_color = (255, 255, 255) if color_name in ['Red', 'Blue', 'Purple', 'Black', 'Brown'] else (0, 0, 0)
        cv2.putText(img_cv, label, (x + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # Draw area info
        area_text = f"{region['area']:.0f} px²"
        cv2.putText(img_cv, area_text, (x + 5, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, boundary_color, 2)
    
    # Convert back to PIL
    img_with_boundaries = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return img_with_boundaries

# Header
st.markdown('<h1 class="main-header">🎨 Color Classification AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict color names from RGB values using Machine Learning</p>', unsafe_allow_html=True)

# Check if model is loaded
if predictor is None:
    st.error("❌ Failed to load the trained model!")
    st.warning("The predictor object is None - model loading returned failure.")
    
    # Show debugging info
    import os
    st.info(f"""
    **Debug Information:**
    - Current directory: `{os.getcwd()}`
    - Model file exists: `{os.path.exists('models/random_forest_color_classifier.pkl')}`
    - Encoder file exists: `{os.path.exists('preprocessed_data/label_encoder.pkl')}`
    
    Check the logs for error details.
    """)
    st.stop()

if not predictor.is_loaded:
    st.error("❌ Model loaded but not marked as ready!")
    st.warning("The predictor was created but is_loaded flag is False.")
    st.info("This might indicate an issue during the model initialization. Check the server logs.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Choose Input Mode:",
        ["🎨 Color Picker", "🎛️ RGB Sliders", "📷 Upload Image", "🔢 Manual RGB Input", "🥽 Mixed Reality AR"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.info(f"**Available Colors:** {len(predictor.get_available_colors())}")
    st.success("**Model:** Random Forest")
    st.success("**Accuracy:** 95.92%")
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("This AI model was trained on RGB color data to predict human-readable color names.")
    
    with st.expander("📋 Available Colors"):
        colors = predictor.get_available_colors()
        for i in range(0, len(colors), 2):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• {colors[i]}")
            if i + 1 < len(colors):
                with col2:
                    st.write(f"• {colors[i+1]}")

# Main content area
st.markdown("---")

# Mode 1: Color Picker
if mode == "🎨 Color Picker":
    st.subheader("🎨 Pick a Color")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        color = st.color_picker("Choose a color:", "#FF0000")
        
        # Convert hex to RGB
        hex_color = color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        st.write(f"**RGB Values:** ({r}, {g}, {b})")
        
        if st.button("🔍 Predict Color", key="predict_picker"):
            with st.spinner('Analyzing color...'):
                color_name, confidence = predictor.predict_with_confidence(r, g, b)
                st.session_state.last_prediction = {
                    'name': color_name,
                    'confidence': confidence,
                    'rgb': (r, g, b)
                }
    
    with col2:
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            
            # Display color box
            st.markdown(f"""
            <div class="color-box" style="background-color: rgb({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]});">
                <div style="background-color: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px;">
                    <p class="prediction-text" style="color: rgb({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]});">
                        {pred['name'].upper()}
                    </p>
                    <p class="confidence-text">
                        Confidence: {pred['confidence']*100:.1f}%
                    </p>
                    <p style="color: #666;">RGB({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Mode 2: RGB Sliders
elif mode == "🎛️ RGB Sliders":
    st.subheader("🎛️ Adjust RGB Values")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        r = st.slider("🔴 Red", 0, 255, 255, key="r_slider")
        g = st.slider("🟢 Green", 0, 255, 0, key="g_slider")
        b = st.slider("🔵 Blue", 0, 255, 0, key="b_slider")
        
        # Auto-predict on slider change
        color_name, confidence = predictor.predict_with_confidence(r, g, b)
        
        st.markdown("---")
        st.write("**Current RGB:**")
        st.code(f"({r}, {g}, {b})")
    
    with col2:
        # Display color box
        st.markdown(f"""
        <div class="color-box" style="background-color: rgb({r}, {g}, {b}); height: 300px;">
            <div style="background-color: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; margin-top: 80px;">
                <p class="prediction-text" style="color: rgb({r}, {g}, {b});">
                    {color_name.upper()}
                </p>
                <p class="confidence-text">
                    Confidence: {confidence*100:.1f}%
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Mode 3: Upload Image
elif mode == "📷 Upload Image":
    st.subheader("📷 Upload an Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        st.markdown("---")
        
        # Option to select number of colors
        col_setting1, col_setting2 = st.columns(2)
        
        with col_setting1:
            num_colors = st.slider("Number of colors to detect:", 2, 10, 5)
        
        with col_setting2:
            analysis_mode = st.radio(
                "Analysis Mode:",
                ["🎨 Dominant Colors", "🔍 Color Boundary Detection", "📊 Color Palette"],
                horizontal=False
            )
        
        # Additional settings for boundary detection
        selected_colors = None
        min_area = 500
        
        if analysis_mode == "🔍 Color Boundary Detection":
            st.markdown("**Select colors to detect:**")
            
            col_a, col_b, col_c = st.columns(3)
            selected_colors = []
            
            with col_a:
                if st.checkbox('Red', value=False, key='chk_red'): selected_colors.append('Red')
                if st.checkbox('Green', value=True, key='chk_green'): selected_colors.append('Green')
                if st.checkbox('Blue', value=False, key='chk_blue'): selected_colors.append('Blue')
                if st.checkbox('Yellow', value=False, key='chk_yellow'): selected_colors.append('Yellow')
            
            with col_b:
                if st.checkbox('Orange', value=False, key='chk_orange'): selected_colors.append('Orange')
                if st.checkbox('Purple', value=False, key='chk_purple'): selected_colors.append('Purple')
                if st.checkbox('Pink', value=False, key='chk_pink'): selected_colors.append('Pink')
                if st.checkbox('Cyan', value=False, key='chk_cyan'): selected_colors.append('Cyan')
            
            with col_c:
                if st.checkbox('White', value=False, key='chk_white'): selected_colors.append('White')
                if st.checkbox('Black', value=False, key='chk_black'): selected_colors.append('Black')
                if st.checkbox('Gray', value=False, key='chk_gray'): selected_colors.append('Gray')
                if st.checkbox('Brown', value=False, key='chk_brown'): selected_colors.append('Brown')
            
            min_area = st.slider("Minimum region size (pixels²):", 100, 5000, 500, 100)
        
        # Analyze button
        button_label = "🔍 Detect Boundaries" if analysis_mode == "🔍 Color Boundary Detection" else "🔍 Analyze Colors"
        
        if st.button(button_label, key="analyze_btn"):
            if analysis_mode == "🔍 Color Boundary Detection":
                if not selected_colors:
                    st.warning("⚠️ Please select at least one color to detect!")
                else:
                    with st.spinner(f'Detecting {", ".join(selected_colors)} regions...'):
                        # Detect color regions
                        detected_regions = detect_color_regions(image, selected_colors, min_area)
                        st.session_state.image_analysis = {
                            'original': image,
                            'mode': analysis_mode,
                            'regions': detected_regions,
                            'selected_colors': selected_colors
                        }
            else:
                with st.spinner(f'Analyzing {num_colors} dominant colors...'):
                    # Extract dominant colors
                    dominant_colors = extract_dominant_colors(image, num_colors)
                    st.session_state.image_analysis = {
                        'original': image,
                        'colors': dominant_colors,
                        'mode': analysis_mode
                    }
        
        # Display results
        if 'image_analysis' in st.session_state:
            results = st.session_state.image_analysis
            
            st.markdown("---")
            
            if results['mode'] == "🎨 Dominant Colors":
                st.subheader("🖼️ Annotated Image")
                
                # Create annotated image
                annotated_img = annotate_image_with_colors(
                    results['original'],
                    results['colors'],
                    predictor
                )
                
                # Display annotated image
                st.image(annotated_img, caption="Image with Color Labels", width='stretch')
                
                # Add download button
                from io import BytesIO
                buf = BytesIO()
                annotated_img.save(buf, format='PNG')
                buf.seek(0)
                
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=buf,
                    file_name="color_analysis.png",
                    mime="image/png"
                )
            
            elif results['mode'] == "🔍 Color Boundary Detection":
                st.subheader("🔍 Color Boundary Detection Results")
                
                detected_regions = results.get('regions', [])
                
                if detected_regions:
                    # Draw boundaries on image
                    boundary_img = draw_color_boundaries(results['original'], detected_regions)
                    
                    # Display image with boundaries
                    st.image(boundary_img, caption="Detected Color Regions", width='stretch')
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Regions", len(detected_regions))
                    
                    with col2:
                        unique_colors = len(set([r['color'] for r in detected_regions]))
                        st.metric("Colors Found", unique_colors)
                    
                    with col3:
                        total_area = sum([r['area'] for r in detected_regions])
                        st.metric("Total Area", f"{total_area:.0f} px²")
                    
                    # Detailed breakdown
                    st.markdown("### Detected Regions Details")
                    
                    # Group by color
                    color_groups = {}
                    for region in detected_regions:
                        color = region['color']
                        if color not in color_groups:
                            color_groups[color] = []
                        color_groups[color].append(region)
                    
                    for color, regions in color_groups.items():
                        with st.expander(f"🎨 {color} - {len(regions)} region(s)"):
                            for i, region in enumerate(regions, 1):
                                x, y, w, h = region['bbox']
                                st.write(f"**Region #{i}**")
                                st.write(f"- Position: ({x}, {y})")
                                st.write(f"- Size: {w} × {h} pixels")
                                st.write(f"- Area: {region['area']:.0f} px²")
                                st.markdown("---")
                    
                    # Download button
                    from io import BytesIO
                    buf = BytesIO()
                    boundary_img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Boundary Detection Image",
                        data=buf,
                        file_name="color_boundaries.png",
                        mime="image/png"
                    )
                else:
                    st.warning(f"⚠️ No {', '.join(results.get('selected_colors', []))} regions found. Try:")
                    st.write("- Selecting different colors")
                    st.write("- Reducing minimum region size")
                    st.write("- Using an image with more distinct colors")
            
            else:  # Color Palette
                st.subheader("🎨 Color Palette")
                
                cols = st.columns(2)
                
                with cols[0]:
                    st.image(results['original'], caption="Original Image", width='stretch')
                
                with cols[1]:
                    st.write("### Detected Colors")
                    
                    for i, (r, g, b, percentage) in enumerate(results['colors'], 1):
                        color_name, confidence = predictor.predict_with_confidence(r, g, b)
                        
                        st.markdown(f"""
                        <div class="color-box" style="background-color: rgb({r}, {g}, {b}); margin: 10px 0; padding: 15px;">
                            <div style="background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px;">
                                <h3 style="color: rgb({r}, {g}, {b}); margin: 0;">#{i} {color_name.upper()}</h3>
                                <p style="color: #666; margin: 5px 0;">RGB({r}, {g}, {b})</p>
                                <p style="color: #666; margin: 5px 0;">Coverage: {percentage:.1f}%</p>
                                <p style="color: #666; margin: 5px 0;">Confidence: {confidence*100:.1f}%</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload an image and click the analyze button to start")

# Mode 4: Manual RGB Input
elif mode == "🔢 Manual RGB Input":
    st.subheader("🔢 Enter RGB Values Manually")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.form("rgb_form"):
            r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
            g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
            b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
            
            submitted = st.form_submit_button("🔍 Predict Color")
            
            if submitted:
                color_name, confidence = predictor.predict_with_confidence(r, g, b)
                st.session_state.manual_prediction = {
                    'name': color_name,
                    'confidence': confidence,
                    'rgb': (r, g, b)
                }
    
    with col2:
        if 'manual_prediction' in st.session_state:
            pred = st.session_state.manual_prediction
            
            st.markdown(f"""
            <div class="color-box" style="background-color: rgb({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]}); height: 300px;">
                <div style="background-color: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; margin-top: 80px;">
                    <p class="prediction-text" style="color: rgb({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]});">
                        {pred['name'].upper()}
                    </p>
                    <p class="confidence-text">
                        Confidence: {pred['confidence']*100:.1f}%
                    </p>
                    <p style="color: #666;">RGB({pred['rgb'][0]}, {pred['rgb'][1]}, {pred['rgb'][2]})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Enter RGB values and click Predict to see results")

# Mode 5: Mixed Reality AR
elif mode == "🥽 Mixed Reality AR":
    st.subheader("🥽 Mixed Reality Color Detection")
    st.markdown("**Real-time AR-style color detection using your webcam**")
    
    # Check if running on Streamlit Cloud
    import socket
    hostname = socket.gethostname()
    is_streamlit_cloud = 'streamlit' in hostname.lower() or os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'
    
    if is_streamlit_cloud:
        st.warning("""
        ⚠️ **WebRTC Camera Access May Not Work on Streamlit Cloud**
        
        Due to network restrictions and firewall configurations, live camera streaming often fails on cloud deployments.
        """)
        
        st.info("""
        ✅ **Recommended Alternatives:**
        
        **Option 1: Use "📷 Upload Image" mode**
        - Upload a photo to detect dominant colors
        - Works perfectly on Streamlit Cloud
        - Provides similar color analysis
        
        **Option 2: Run locally for guaranteed camera access**
        ```bash
        git clone https://github.com/tharungvs25/color_classification.git
        cd color_classification
        pip install -r requirements.txt
        streamlit run app.py
        ```
        
        **Option 3: Try anyway (may work on some networks)**
        - Click START below and wait 30 seconds
        - If connection fails, use Upload Image mode instead
        """)
        
        if not st.checkbox("⚠️ I understand - Try WebRTC anyway", key="acknowledge_webrtc"):
            st.stop()
        
        st.markdown("---")
    
    st.info("""
    📹 **How to use:**
    1. Click "START" below to activate your camera
    2. Allow camera permissions when your browser asks
    3. Wait up to 30 seconds for connection
    4. Point your camera at objects to detect colors in real-time
    """)
    
    # Video transformer class
    class ColorDetectionTransformer(VideoTransformerBase):
        def __init__(self):
            self.predictor = predictor
            
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Get center color
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Sample 5x5 region at center
            region = img[max(0, center_y-2):min(height, center_y+3), 
                        max(0, center_x-2):min(width, center_x+3)]
            
            # Get average color (OpenCV uses BGR)
            b, g, r = cv2.mean(region)[:3]
            
            # Predict color
            try:
                color_name, confidence = self.predictor.predict_with_confidence(
                    int(r), int(g), int(b)
                )
            except:
                color_name = "Unknown"
                confidence = 0.0
            
            # Draw center crosshair
            cv2.circle(img, (center_x, center_y), 30, (0, 255, 255), 2)
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
            cv2.line(img, (center_x - 40, center_y), (center_x + 40, center_y), (0, 255, 255), 2)
            cv2.line(img, (center_x, center_y - 40), (center_x, center_y + 40), (0, 255, 255), 2)
            
            # Create label
            label = f"{color_name.upper()}"
            conf_label = f"{confidence*100:.0f}%"
            rgb_label = f"RGB({int(r)},{int(g)},{int(b)})"
            
            # Draw semi-transparent background for text
            overlay = img.copy()
            cv2.rectangle(overlay, (center_x - 150, center_y - 120), 
                         (center_x + 150, center_y - 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Draw border
            cv2.rectangle(img, (center_x - 150, center_y - 120), 
                         (center_x + 150, center_y - 40), (0, 255, 255), 2)
            
            # Draw text
            cv2.putText(img, label, (center_x - 140, center_y - 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, rgb_label, (center_x - 140, center_y - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img, conf_label, (center_x - 140, center_y - 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return img
    
    # WebRTC Configuration with multiple STUN servers for better connectivity
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]}
    )
    
    st.info("""
    ⏳ **First time setup may take 10-30 seconds**
    - Allow camera permissions when prompted
    - Wait for the connection to establish
    - If it takes too long, try refreshing the page
    """)
    
    # Start webcam stream with error handling
    try:
        webrtc_ctx = webrtc_streamer(
            key="color-detection",
            video_transformer_factory=ColorDetectionTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception as e:
        st.error(f"⚠️ Failed to initialize camera: {str(e)}")
        st.info("Try refreshing the page or use the **Upload Image** mode as an alternative.")
        webrtc_ctx = None
    
    st.markdown("---")
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting Connection Issues"):
        st.markdown("""
        **If the camera connection is taking too long:**
        
        1. **Refresh the page** - Sometimes a fresh start helps
        2. **Check browser permissions** - Make sure your browser has camera access
        3. **Try a different browser** - Chrome and Edge work best
        4. **Check your network** - Firewall or corporate network may block WebRTC
        5. **Use a different mode** - Try the "Upload Image" mode as an alternative
        
        **Supported browsers:**
        - ✅ Google Chrome (Recommended)
        - ✅ Microsoft Edge
        - ✅ Firefox
        - ⚠️ Safari (may have limited support)
        
        **If issues persist:**
        - The other 4 modes (Color Picker, RGB Sliders, Upload Image, Manual Input) work perfectly!
        - Download and run locally: `streamlit run app.py` for guaranteed camera access
        """)
    
    st.markdown("""
    ### 🎮 How It Works:
    - **Crosshair**: Shows the detection point (center of frame)
    - **Color Label**: Displays the detected color name
    - **RGB Values**: Shows the exact RGB components
    - **Confidence**: Model's confidence in the prediction
    
    ### 💡 Tips:
    - Ensure good lighting for better detection
    - Hold camera steady for clear results
    - Point the center crosshair at the object you want to identify
    - Works on both local and cloud deployments!
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Quick Stats")
    st.write(f"**Model Type:** Random Forest")
    st.write(f"**Accuracy:** 95.92%")

with col2:
    st.markdown("### 🎯 Common Colors")
    st.write("Red • Green • Blue • Yellow")
    st.write("Orange • Purple • Pink • Brown")

with col3:
    st.markdown("### 💡 Tips")
    st.write("• Use sliders for live preview")
    st.write("• Upload images to analyze")
    st.write("• Try different RGB combinations")

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit & Scikit-learn • Machine Learning Project 2026</p>',
    unsafe_allow_html=True
)
