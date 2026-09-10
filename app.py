# app.py
import time
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# Configuration & Class Labels
CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)

st.set_page_config(page_title="CIFAR-10 Mini-ResNet Engine", layout="wide")

@st.cache_resource
def load_session(onnx_path="./artifacts/cifar10_miniresnet.onnx"):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, options, providers=['CPUExecutionProvider'])

session = load_session()
input_name = session.get_inputs()[0].name

st.title("Edge-Optimized Mini-ResNet Inference Engine")
st.markdown("Production inference runtime powered by **ONNX Runtime (CPUExecutionProvider)**.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Pipeline")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        st.image(raw_img, caption="Uploaded Image", use_container_width=True)
        
        # Preprocessing: Resize to 32x32 -> Normalize to (3, 32, 32)
        resized_img = raw_img.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.array(resized_img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
        arr = (arr - CIFAR10_MEAN) / CIFAR10_STD
        input_tensor = np.expand_dims(arr, axis=0)  # Shape: (1, 3, 32, 32)

with col2:
    st.subheader("Inference & Telemetry")
    if uploaded_file:
        # Run inference and measure runtime
        t0 = time.perf_counter()
        raw_outputs = session.run(None, {input_name: input_tensor})[0]
        inference_latency = (time.perf_counter() - t0) * 1000  # ms
        
        # Softmax for probabilities
        exp_vals = np.exp(raw_outputs - np.max(raw_outputs))
        probabilities = (exp_vals / np.sum(exp_vals))[0]
        
        top3_indices = np.argsort(probabilities)[::-1][:3]
        predicted_class = CLASSES[top3_indices[0]]
        confidence = probabilities[top3_indices[0]] * 100

        st.metric("Top Prediction", predicted_class, f"{confidence:.2f}% Confidence")
        st.metric("Hardware Latency", f"{inference_latency:.3f} ms", "ONNX CPU")

        st.markdown("#### Probability Distribution")
        for idx in top3_indices:
            st.progress(float(probabilities[idx]), text=f"{CLASSES[idx]}: {probabilities[idx]*100:.2f}%")