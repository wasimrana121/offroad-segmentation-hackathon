# 🏜️ Offroad Semantic Segmentation App

A Streamlit-based web application for semantic segmentation of desert environments using deep learning.

## 🎯 Features

- **Image Segmentation**: Upload and segment single desert images
- **Video Segmentation**: Process entire videos frame-by-frame
- **Real-time Visualization**: View segmentation masks and overlays
- **Performance Metrics**: Display model performance and per-class IoU scores
- **Interactive Controls**: Adjust transparency, choose visualization modes
- **Download Results**: Export segmented images and videos

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster processing)
- 4GB+ RAM (8GB+ recommended for video processing)

## 🚀 Installation

### 1. Clone or Download this project

```bash
# Create a project directory
mkdir segmentation-app
cd segmentation-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Model

Place your trained model file in the project directory:

```
segmentation-app/
├── app.py
├── requirements.txt
├── best_model.pth  ← Your trained model file
└── README.md
```

## 🏃 Running the App

### Local Development

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Running on a Specific Port

```bash
streamlit run app.py --server.port 8080
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Free Hosting)

1. Push your code to GitHub (exclude model file if >100MB)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

**For large model files (>100MB):**
- Use Git LFS or
- Host model on Google Drive/Dropbox and modify code to download it
- Use Hugging Face Hub to host your model

### Option 2: Hugging Face Spaces

1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new Space with Streamlit
3. Upload your files
4. Add `best_model.pth` to the space
5. Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/APP_NAME`

### Option 3: Railway.app

1. Create account at [railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Add Procfile:
```
web: streamlit run app.py --server.port $PORT
```
4. Deploy!

### Option 4: Heroku

1. Create account at [heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Create these files:

**Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

4. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 5: AWS EC2 / Google Cloud / Azure

1. Launch a VM instance
2. Install Python and dependencies
3. Run with:
```bash
nohup streamlit run app.py --server.port 80 &
```
4. Configure security groups to allow HTTP/HTTPS traffic

## 📁 Project Structure

```
segmentation-app/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── best_model.pth         # Your trained model (you add this)
├── README.md              # This file
│
└── (optional files for deployment)
    ├── Procfile           # For Heroku
    ├── setup.sh          # For Heroku
    └── .streamlit/       # Streamlit config
        └── config.toml
```

## 🎨 Using the App

### Image Segmentation

1. Go to the **"📷 Image Segmentation"** tab
2. Upload a desert image (JPG, PNG, JPEG)
3. Adjust visualization options:
   - Toggle segmentation mask
   - Toggle overlay view
   - Adjust transparency
4. Click **"🚀 Run Segmentation"**
5. Download results using the download buttons

### Video Segmentation

1. Go to the **"🎥 Video Segmentation"** tab
2. Upload a video file (MP4, AVI, MOV)
3. Adjust processing options:
   - Frame processing interval
   - Overlay transparency
4. Click **"🎬 Process Video"**
5. Wait for processing (progress bar shows status)
6. Download the segmented video

## 🔧 Customization

### Modify Model Loading

If your model has a different structure, edit the `load_model()` function in `app.py`:

```python
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example: Load a specific architecture
    # model = YourModelClass()
    # model.load_state_dict(torch.load(model_path))
    
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device
```

### Change Input Size

Modify the `preprocess_image()` function:

```python
transforms.Resize((512, 512)),  # Change to your model's input size
```

### Add New Classes

Update the `CLASSES` dictionary with your class definitions:

```python
CLASSES = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    # Add your classes here
}
```

## 📊 Model Performance

Current model metrics (from `best_model.pth`):

- **Mean IoU**: 52.57%
- **Validation IoU**: 52.57%
- **Best Epoch**: 4
- **TTA**: Enabled

### Per-Class Performance:
- Sky: 96.26%
- Landscape: 58.77%
- Dry Grass: 43.87%
- Trees: 28.63%
- Dry Bushes: 23.44%
- Rocks: 5.59%

## 🐛 Troubleshooting

### "Model file not found"
- Ensure `best_model.pth` is in the same directory as `app.py`
- Check file permissions

### Out of Memory (OOM) errors
- Reduce batch size or image size
- Process videos with lower frame rates
- Use CPU instead of GPU for smaller workloads

### Slow video processing
- Process every Nth frame instead of every frame
- Use a more powerful GPU
- Reduce video resolution before processing

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

## 📝 License

This project was created for the Duality AI Hackathon Challenge.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📧 Contact

For questions about this project, please refer to the Duality AI Discord community.

---

**Built with ❤️ for the Duality AI Offroad Autonomy Segmentation Challenge**
