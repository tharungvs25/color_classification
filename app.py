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
import os

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

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and encoder."""
    predictor = ColorPredictor()
    if predictor.load_model():
        return predictor
    return None

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

# Initialize predictor
if st.session_state.predictor is None:
    with st.spinner('Loading AI model...'):
        st.session_state.predictor = load_model()
        if st.session_state.predictor is not None:
            st.session_state.model_loaded = True

# Header
st.markdown('<h1 class="main-header">🎨 Color Classification AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict color names from RGB values using Machine Learning</p>', unsafe_allow_html=True)

# Check if model is loaded
if not st.session_state.model_loaded:
    st.error("❌ Model not loaded! Please run `python train_model.py` first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Choose Input Mode:",
        ["🎨 Color Picker", "🎛️ RGB Sliders", "📷 Upload Image", "🔢 Manual RGB Input"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📊 Model Info")
    st.info(f"**Available Colors:** {len(st.session_state.predictor.get_available_colors())}")
    st.success("**Model:** Random Forest")
    st.success("**Accuracy:** 95.92%")
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("This AI model was trained on RGB color data to predict human-readable color names.")
    
    with st.expander("📋 Available Colors"):
        colors = st.session_state.predictor.get_available_colors()
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
                color_name, confidence = st.session_state.predictor.predict_with_confidence(r, g, b)
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
        color_name, confidence = st.session_state.predictor.predict_with_confidence(r, g, b)
        
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
                ["🎨 Dominant Colors (Annotated)", "📊 Color Palette Only"],
                horizontal=True
            )
        
        if st.button("🔍 Analyze Colors", key="analyze_btn"):
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
            
            if results['mode'] == "🎨 Dominant Colors (Annotated)":
                st.subheader("🖼️ Annotated Image")
                
                # Create annotated image
                annotated_img = annotate_image_with_colors(
                    results['original'],
                    results['colors'],
                    st.session_state.predictor
                )
                
                # Display annotated image
                st.image(annotated_img, caption="Image with Color Labels", width='stretch')
                
                # Add download button
                # Convert to bytes for download
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
            
            else:  # Color Palette Only
                st.subheader("🎨 Color Palette")
                
                cols = st.columns(2)
                
                with cols[0]:
                    st.image(results['original'], caption="Original Image", width='stretch')
                
                with cols[1]:
                    st.write("### Detected Colors")
                    
                    for i, (r, g, b, percentage) in enumerate(results['colors'], 1):
                        color_name, confidence = st.session_state.predictor.predict_with_confidence(r, g, b)
                        
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
            st.info("👆 Upload an image and click 'Analyze Colors' to detect dominant colors")

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
                color_name, confidence = st.session_state.predictor.predict_with_confidence(r, g, b)
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
