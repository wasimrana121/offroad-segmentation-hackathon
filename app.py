import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import json
import tempfile
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Offroad Semantic Segmentation",
    page_icon="🏜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .class-legend {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    h1 {
        color: #667eea;
    }
    .stAlert {
        background-color: #e8f4f8;
        border: 1px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Class definitions
CLASSES = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    100: {'name': 'Trees', 'color': (34, 139, 34)},
    200: {'name': 'Lush Bushes', 'color': (144, 238, 144)},
    300: {'name': 'Dry Grass', 'color': (244, 164, 96)},
    500: {'name': 'Dry Bushes', 'color': (210, 105, 30)},
    550: {'name': 'Ground Clutter', 'color': (160, 82, 45)},
    600: {'name': 'Flowers', 'color': (255, 105, 180)},
    700: {'name': 'Logs', 'color': (139, 69, 19)},
    800: {'name': 'Rocks', 'color': (128, 128, 128)},
    7100: {'name': 'Landscape', 'color': (222, 184, 135)},
    10000: {'name': 'Sky', 'color': (135, 206, 235)}
}

# Model metrics
MODEL_METRICS = {
    'mean_iou': 0.5257,
    'val_iou': 0.5257,
    'epoch': 4,
    'use_tta': True,
    'per_class_iou': {
        'Trees': 0.2863,
        'Lush Bushes': 0.0014,
        'Dry Grass': 0.4387,
        'Dry Bushes': 0.2344,
        'Ground Clutter': 0.0,
        'Rocks': 0.0559,
        'Landscape': 0.5877,
        'Sky': 0.9626
    }
}

@st.cache_resource
def load_model(model_path):
    """Load the trained segmentation model"""
    try:
        # Load model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_color_map():
    """Create color map for visualization"""
    color_map = np.zeros((11000, 3), dtype=np.uint8)
    for class_id, class_info in CLASSES.items():
        if class_id < 11000:
            color_map[class_id] = class_info['color']
    return color_map

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert PIL Image to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Adjust based on your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_segmentation(model, image, device):
    """Run segmentation prediction"""
    try:
        with torch.no_grad():
            image_tensor = preprocess_image(image).to(device)
            output = model(image_tensor)
            
            # Get predicted class for each pixel
            if isinstance(output, dict):
                output = output['out']
            
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
            return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def visualize_segmentation(prediction, original_image, alpha=0.6):
    """Create visualization of segmentation results"""
    color_map = create_color_map()
    
    # Resize prediction to match original image size
    original_size = original_image.size
    prediction_resized = cv2.resize(prediction.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    # Create colored segmentation mask
    colored_mask = color_map[prediction_resized]
    
    # Convert original image to numpy array
    original_np = np.array(original_image)
    
    # Create overlay
    overlay = cv2.addWeighted(original_np, 1-alpha, colored_mask, alpha, 0)
    
    return colored_mask, overlay

def process_video(video_path, model, device, progress_bar, status_text):
    """Process video frame by frame"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run segmentation
        prediction = predict_segmentation(model, pil_image, device)
        
        if prediction is not None:
            # Create visualization
            _, overlay = visualize_segmentation(prediction, pil_image, alpha=0.5)
            
            # Convert back to BGR for video writing
            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(overlay_bgr)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏜️ Offroad Semantic Segmentation")
        st.markdown("**AI-Powered Desert Environment Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        # Model metrics display
        st.markdown("### Performance Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <h2>{MODEL_METRICS['mean_iou']*100:.2f}%</h2>
            <p>Mean IoU Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation IoU", f"{MODEL_METRICS['val_iou']*100:.2f}%")
        with col2:
            st.metric("Epoch", MODEL_METRICS['epoch'])
        
        st.info(f"🔬 Test-Time Augmentation: {'Enabled' if MODEL_METRICS['use_tta'] else 'Disabled'}")
        
        st.markdown("---")
        
        # Class legend
        st.markdown("### 🎨 Class Legend")
        st.markdown('<div class="class-legend">', unsafe_allow_html=True)
        
        for class_id, class_info in CLASSES.items():
            if class_id != 0:  # Skip background
                color_hex = '#{:02x}{:02x}{:02x}'.format(*class_info['color'])
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color_hex}; 
                                border-radius: 4px; margin-right: 10px; border: 1px solid #ccc;"></div>
                    <span style="font-size: 14px;">{class_info['name']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Per-class performance
        with st.expander("📈 Per-Class IoU Scores"):
            for class_name, iou in MODEL_METRICS['per_class_iou'].items():
                st.progress(iou, text=f"{class_name}: {iou*100:.2f}%")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Segmentation", "🎥 Video Segmentation", "ℹ️ About"])
    
    # Image Segmentation Tab
    with tab1:
        st.header("Upload and Segment Images")
        
        uploaded_file = st.file_uploader(
            "Choose a desert image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a desert environment for segmentation"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Segmentation options
            st.markdown("### Visualization Options")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                show_mask = st.checkbox("Show Segmentation Mask", value=True)
            with col_opt2:
                show_overlay = st.checkbox("Show Overlay", value=True)
            with col_opt3:
                alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.6, 0.1)
            
            # Run segmentation button
            if st.button("🚀 Run Segmentation", type="primary"):
                with st.spinner("Loading model..."):
                    # Try to load the model
                    model_path = "best_model.pth"
                    if os.path.exists(model_path):
                        model, device = load_model(model_path)
                    else:
                        st.warning("Model file not found. Using demo visualization.")
                        model, device = None, None
                
                with st.spinner("Processing image..."):
                    if model is not None:
                        # Real prediction
                        prediction = predict_segmentation(model, image, device)
                        
                        if prediction is not None:
                            colored_mask, overlay = visualize_segmentation(prediction, image, alpha)
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            if show_mask:
                                with col1:
                                    st.subheader("Segmentation Mask")
                                    st.image(colored_mask, use_container_width=True)
                            
                            if show_overlay:
                                with col2:
                                    st.subheader("Overlay View")
                                    st.image(overlay, use_container_width=True)
                            
                            st.success("✅ Segmentation completed successfully!")
                            
                            # Download button
                            col1, col2 = st.columns(2)
                            with col1:
                                # Convert mask to bytes for download
                                mask_img = Image.fromarray(colored_mask.astype(np.uint8))
                                mask_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                                mask_img.save(mask_bytes.name)
                                with open(mask_bytes.name, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Mask",
                                        data=f,
                                        file_name="segmentation_mask.png",
                                        mime="image/png"
                                    )
                            
                            with col2:
                                overlay_img = Image.fromarray(overlay.astype(np.uint8))
                                overlay_bytes = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                                overlay_img.save(overlay_bytes.name)
                                with open(overlay_bytes.name, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Overlay",
                                        data=f,
                                        file_name="segmentation_overlay.png",
                                        mime="image/png"
                                    )
                    else:
                        # Demo mode - show original image
                        st.info("💡 Demo Mode: Add 'best_model.pth' to the app directory for real predictions")
                        with col2:
                            st.subheader("Demo Output")
                            st.image(image, use_container_width=True)
    
    # Video Segmentation Tab
    with tab2:
        st.header("Upload and Segment Videos")
        st.info("⚠️ Video processing may take several minutes depending on video length and your hardware.")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video of a desert environment for segmentation"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            # Display original video
            st.subheader("Original Video")
            st.video(video_path)
            
            # Video processing options
            col1, col2 = st.columns(2)
            with col1:
                process_every_n_frames = st.number_input(
                    "Process every N frames (1 = every frame)",
                    min_value=1,
                    max_value=30,
                    value=1,
                    help="Processing every frame is slower but more accurate"
                )
            
            with col2:
                overlay_alpha_video = st.slider(
                    "Video Overlay Transparency",
                    0.0, 1.0, 0.5, 0.1
                )
            
            # Process video button
            if st.button("🎬 Process Video", type="primary"):
                model_path = "best_model.pth"
                
                if os.path.exists(model_path):
                    with st.spinner("Loading model..."):
                        model, device = load_model(model_path)
                    
                    if model is not None:
                        st.subheader("Processing Progress")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            output_video_path = process_video(
                                video_path, model, device, 
                                progress_bar, status_text
                            )
                            
                            st.success("✅ Video processing completed!")
                            
                            # Display processed video
                            st.subheader("Segmented Video")
                            st.video(output_video_path)
                            
                            # Download button
                            with open(output_video_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=f,
                                    file_name="segmented_video.mp4",
                                    mime="video/mp4"
                                )
                            
                            # Cleanup
                            os.unlink(output_video_path)
                        
                        except Exception as e:
                            st.error(f"Error processing video: {str(e)}")
                else:
                    st.error("❌ Model file 'best_model.pth' not found. Please add the model file to run predictions.")
            
            # Cleanup temp video file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    # About Tab
    with tab3:
        st.header("About This Project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            This is a semantic segmentation model trained for the **Duality AI Offroad Autonomy 
            Segmentation Challenge**. The model analyzes desert environments and classifies 
            every pixel into one of 11 categories.
            
            ### 🏗️ Model Architecture
            - **Architecture**: Deep Learning Segmentation Model
            - **Training Data**: Synthetic desert environments from FalconEditor
            - **Input Size**: Variable (resized to model input)
            - **Output**: Per-pixel class predictions
            
            ### 📊 Training Details
            - **Best Epoch**: 4
            - **Mean IoU**: 52.57%
            - **Validation IoU**: 52.57%
            - **Test-Time Augmentation**: Enabled
            """)
        
        with col2:
            st.markdown("""
            ### 🎨 Segmentation Classes
            The model can identify the following terrain features:
            
            1. **Trees** (34.87% IoU) - Desert trees and large vegetation
            2. **Lush Bushes** (0.14% IoU) - Green bushes and shrubs
            3. **Dry Grass** (43.87% IoU) - Dried grass and low vegetation
            4. **Dry Bushes** (23.44% IoU) - Dead or dry shrubs
            5. **Ground Clutter** (0% IoU) - Small debris on ground
            6. **Flowers** - Desert flowers and blooms
            7. **Logs** - Fallen wood and tree trunks
            8. **Rocks** (5.59% IoU) - Rocks and boulders
            9. **Landscape** (58.77% IoU) - General terrain/ground
            10. **Sky** (96.26% IoU) - Sky and clouds
            
            ### 🚀 Use Cases
            - Autonomous navigation in desert terrains
            - Obstacle detection for UGVs
            - Path planning for offroad vehicles
            - Environmental analysis and mapping
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Technical Details
        
        **Digital Twins for Segmentation**: This project demonstrates the power of synthetic 
        data generation using digital twin technology. All training data was generated from 
        Duality AI's Falcon platform, which creates geospatially-accurate virtual 
        environments of desert terrains.
        
        **Benefits of Synthetic Data**:
        - Cost-effective data generation
        - Perfect pixel-level annotations
        - Control over environmental conditions
        - Rapid dataset creation and iteration
        
        ### 🔗 Resources
        - [Duality AI](https://duality.ai)
        - [Falcon Platform](https://falcon.duality.ai)
        - Challenge Documentation (see uploaded PDF)
        """)
        
        st.markdown("---")
        
        st.info("""
        **Note**: To use this app with real predictions, place your trained model file 
        named `best_model.pth` in the same directory as this application.
        """)

if __name__ == "__main__":
    main()