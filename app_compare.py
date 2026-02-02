import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Rice Disease Classifier", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Rice Leaf Disease Recognition")
st.caption("Powered by Advanced CNN Models")

@st.cache_resource
def load_best_model():
    model_path = 'best_tuned_model.keras'
    if not os.path.exists(model_path):
        return None, None
    
    try:
        # Attempt standard load
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as e:
        # Fallback for quantization config issue if it reappears
        st.error(f"Error loading model: {e}")
        return None, None

    info = {}
    if os.path.exists('best_tuned_model_info.json'):
        with open('best_tuned_model_info.json', 'r') as f:
            info = json.load(f)
            
    return model, info

if not os.path.exists('best_tuned_model_info.json'):
    st.warning("⚠️ Optimization is still in progress. Please wait for `train_optimized.py` to finish.")
    if st.button("Refresh status"):
        st.experimental_rerun()
    st.stop()

model, model_info = load_best_model()

if model is None:
    st.warning("⚠️ Model file not found. Please ensure training is complete.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2518/2518048.png", width=100)
    st.header("Model Details")
    if model_info:
        st.success(f"**Architecture:** {model_info.get('model_name', 'Unknown')}")
        st.info(f"**Validation Accuracy:** {model_info.get('accuracy', 0.0):.4f}")
    else:
        st.info("Training metadata not available.")
    
    st.divider()
    st.write("Supported Classes:")
    class_names = model_info.get('class_names', ['blast', 'healthy', 'insect', 'leaf_folder', 'scald', 'stripes', 'tungro'])
    for c in class_names:
        st.markdown(f"- *{c}*")

st.write("### Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            # The model includes a Rescaling layer, so we passed 0-255 input (which is what standard loading does).
            # img_to_array returns float32 0-255? Yes.
            
            predictions = model.predict(img_array)
            # predictions are already probabilities due to softmax in model
            
            confidence = np.max(predictions[0])
            class_idx = np.argmax(predictions[0])
            
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
            else:
                class_name = f"Unknown ({class_idx})"
            
            # Display Result
            st.markdown(f"""
            <div class="metric-container">
                <h2>Prediction: {class_name}</h2>
                <h1>{confidence*100:.2f}% Confidence</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart for top k?
            st.bar_chart(predictions[0])
