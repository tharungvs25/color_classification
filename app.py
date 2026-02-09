"""
Color Classification Web UI
============================
Interactive Streamlit web application for color prediction.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
from predict_color import ColorPredictor
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
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Option to select detection area
            detection_mode = st.radio(
                "Pixel Selection:",
                ["Center Pixel", "Average Color", "Click to Select"]
            )
        
        with col2:
            if detection_mode == "Center Pixel":
                # Get center pixel
                h, w = img_array.shape[:2]
                center_pixel = img_array[h//2, w//2]
                
                if len(center_pixel) >= 3:
                    r, g, b = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])
                else:
                    st.error("Image must be in RGB format")
                    st.stop()
                
            elif detection_mode == "Average Color":
                # Calculate average color
                avg_color = img_array.mean(axis=(0, 1))
                r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
            
            else:
                st.info("Click mode coming soon! Using center pixel for now.")
                h, w = img_array.shape[:2]
                center_pixel = img_array[h//2, w//2]
                r, g, b = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])
            
            # Predict
            color_name, confidence = st.session_state.predictor.predict_with_confidence(r, g, b)
            
            # Display result
            st.markdown(f"""
            <div class="color-box" style="background-color: rgb({r}, {g}, {b});">
                <div style="background-color: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px;">
                    <p class="prediction-text" style="color: rgb({r}, {g}, {b});">
                        {color_name.upper()}
                    </p>
                    <p class="confidence-text">
                        Confidence: {confidence*100:.1f}%
                    </p>
                    <p style="color: #666;">RGB({r}, {g}, {b})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
