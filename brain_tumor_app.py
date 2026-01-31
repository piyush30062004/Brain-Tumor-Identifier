import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional desktop-like appearance
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Main content container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Section */
    .app-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .app-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .app-subtitle {
        text-align: center;
        font-size: 18px;
        color: #b0b0b0;
        margin-top: 10px;
        font-weight: 400;
    }
    
    /* Card Styles */
    .card {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #333333 0%, #1e1e1e 100%);
    }
    
    /* Result Cards */
    .result-card-tumor {
        background: linear-gradient(135deg, #ff4b4b 0%, #c92a2a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255,75,75,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        animation: slideIn 0.5s ease;
    }
    
    .result-card-normal {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(81,207,102,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-icon {
        font-size: 64px;
        margin-bottom: 1rem;
    }
    
    .result-title {
        font-size: 36px;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .result-confidence {
        font-size: 20px;
        font-weight: 500;
        opacity: 0.95;
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stat-label {
        font-size: 14px;
        color: #b0b0b0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #1a1a1a;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(102,126,234,0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box p {
        margin: 0;
        color: #e0e0e0;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Warning Box */
    .warning-box {
        background: rgba(255,193,7,0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box p {
        margin: 0;
        color: #ffc107;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Image container */
    .image-container {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'tumor_detected' not in st.session_state:
    st.session_state.tumor_detected = 0

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI")
    st.markdown("---")
    
    st.markdown("#### 📊 Session Statistics")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Scans</div>
        <div class="stat-value">{st.session_state.predictions_made}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Tumors Detected</div>
        <div class="stat-value">{st.session_state.tumor_detected}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### ℹ️ About")
    st.markdown("""
    <div class="info-box">
        <p><b>NeuroScan AI</b> uses advanced deep learning algorithms to analyze brain MRI scans and detect potential tumor presence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Model Information")
    st.markdown("""
    - **Architecture:** CNN (Convolutional Neural Network)
    - **Input Size:** 224x224 pixels
    - **Model Type:** Binary Classification
    - **Framework:** TensorFlow/Keras
    """)
    
    st.markdown("---")
    
    st.markdown("#### ⚠️ Disclaimer")
    st.markdown("""
    <div class="warning-box">
        <p>This tool is for educational purposes only. Always consult with qualified medical professionals for diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"<small>© 2024 NeuroScan AI | v1.0.0</small>", unsafe_allow_html=True)

# Load model function with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("brain_tumor_model.keras")
        return model
    except Exception as e:
        # This prevents the app from crashing if the file is missing
        st.sidebar.warning("⚠️ Model file not found. Running in Demo Mode.")
        return "DEMO"
# Main content
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🧠 NeuroScan AI</h1>
        <p class="app-subtitle">Advanced Brain Tumor Detection System powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📤 Upload MRI Scan</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is None:
            st.markdown("""
            <div class="upload-section">
                <div style="font-size: 64px; margin-bottom: 1rem;">📁</div>
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">Drop your MRI scan here</h3>
                <p style="color: #b0b0b0;">Supported formats: JPG, JPEG, PNG</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            try:
                st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
            except TypeError:
                st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image info
            st.markdown(f"""
            <div class="info-box">
                <p><b>Image Details:</b> {image.size[0]}x{image.size[1]} pixels | Format: {image.format}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("""
            <div class="card">
                <div class="card-title">🔬 Analysis Results</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Load model
            model = load_model()
            
            if model is not None:
                with st.spinner("🔄 Analyzing MRI scan..."):
                    # 1. Preprocess image
                    img = np.array(image)
                    
                    # Convert to RGB (standard for 224x224x3 models)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    elif img.shape[2] == 1: # Handle single channel arrays
                        img = cv2.merge([img, img, img])
                    
                    # 2. Resize and normalize
                    img = cv2.resize(img, (224, 224))
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=0) # Better than .reshape()
                    
                    # 3. Prediction Handling
                    if model == "DEMO":
                        import time
                        time.sleep(1) # Simulate AI processing
                        pred = 0.87 # Static value for testing UI, or use np.random.rand()
                    else:
                        prediction = model.predict(img, verbose=0)
                        # Extract single value regardless of output shape [1,1] or [1,]
                        pred = float(prediction[0][0] if len(prediction.shape) > 1 else prediction[0])
                    
                    # 4. Update session state (Use a trigger so it doesn't loop)
                    if 'last_analyzed' not in st.session_state or st.session_state.last_analyzed != uploaded_file.name:
                        st.session_state.predictions_made += 1
                        if pred > 0.5:
                            st.session_state.tumor_detected += 1
                        st.session_state.last_analyzed = uploaded_file.name
                # Display results
                if pred > 0.5:
                    st.markdown(f"""
                    <div class="result-card-tumor">
                        <div class="result-icon">🚨</div>
                        <div class="result-title">TUMOR DETECTED</div>
                        <div class="result-confidence">Confidence: {pred*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <p><b>⚠️ Important:</b> A potential tumor has been detected. Please consult with a neurologist or oncologist immediately for professional evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-normal">
                        <div class="result-icon">✅</div>
                        <div class="result-title">NO TUMOR DETECTED</div>
                        <div class="result-confidence">Confidence: {(1-pred)*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <p><b>ℹ️ Note:</b> No tumor detected in this scan. However, regular medical check-ups are always recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence gauge chart
                st.markdown("#### 📊 Confidence Visualization")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred * 100 if pred > 0.5 else (1-pred) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction Confidence", 'font': {'size': 24, 'color': '#ffffff'}},
                    number = {'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                        'bar': {'color': "#ff4b4b" if pred > 0.5 else "#51cf66"},
                        'bgcolor': "#2d2d2d",
                        'borderwidth': 2,
                        'bordercolor': "#ffffff",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(81, 207, 102, 0.3)'},
                            {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.3)'},
                            {'range': [75, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="#1e1e1e",
                    plot_bgcolor="#1e1e1e",
                    font={'color': "#ffffff", 'family': "Inter"},
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"""
                <div style="text-align: center; color: #b0b0b0; margin-top: 2rem; font-size: 14px;">
                    Analysis completed at: {timestamp}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 64px; margin-bottom: 1rem; opacity: 0.5;">🔬</div>
                <h3 style="color: #b0b0b0; font-weight: 400;">Upload an MRI scan to begin analysis</h3>
            </div>
            """, unsafe_allow_html=True)

    # How it works section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-title">🔍 How It Works</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 48px; margin-bottom: 1rem;">📤</div>
            <h4 style="color: #667eea;">1. Upload</h4>
            <p style="color: #b0b0b0; font-size: 14px;">Upload your brain MRI scan image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 48px; margin-bottom: 1rem;">🔄</div>
            <h4 style="color: #667eea;">2. Process</h4>
            <p style="color: #b0b0b0; font-size: 14px;">AI processes and analyzes the scan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 48px; margin-bottom: 1rem;">🧠</div>
            <h4 style="color: #667eea;">3. Detect</h4>
            <p style="color: #b0b0b0; font-size: 14px;">Deep learning model detects tumors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 48px; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #667eea;">4. Results</h4>
            <p style="color: #b0b0b0; font-size: 14px;">Get instant results with confidence</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
