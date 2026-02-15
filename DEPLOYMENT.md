# 🚀 Deployment Guide for Streamlit Cloud

## Quick Deploy to Streamlit Cloud (Free!)

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** for your project
2. **Upload these files** to your repository:
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   └── .streamlit/
       └── config.toml
   ```

3. **For the model file (`best_model.pth`)** - Since it's 29MB:
   - You can upload it directly to GitHub (files up to 100MB are allowed)
   - Or use Git LFS for large files
   - Or host it externally (see alternatives below)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository
5. Choose branch: `main` (or `master`)
6. Main file path: `app.py`
7. Click **"Deploy!"**

Your app will be live at: `https://your-username-your-repo-name-xxx.streamlit.app`

---

## Alternative: Host Model on External Storage

If your model is too large (>100MB), you can host it externally and download it at runtime.

### Option A: Google Drive

1. Upload `best_model.pth` to Google Drive
2. Make it publicly accessible
3. Get the file ID from the shareable link
4. Modify `app.py` to download the model:

```python
import gdown

@st.cache_resource
def load_model(model_path):
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (this may take a minute)"):
            url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
            gdown.download(url, model_path, quiet=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device
```

Add to `requirements.txt`:
```
gdown==4.7.1
```

### Option B: Hugging Face Hub

1. Create account at [huggingface.co](https://huggingface.co)
2. Upload model to Hub:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_file(path_or_fileobj='best_model.pth', path_in_repo='best_model.pth', repo_id='YOUR_USERNAME/offroad-segmentation', repo_type='model')"
```

3. Modify `app.py`:

```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/offroad-segmentation",
        filename="best_model.pth"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device
```

Add to `requirements.txt`:
```
huggingface_hub==0.20.0
```

### Option C: Dropbox

1. Upload model to Dropbox
2. Get public link
3. Convert link: Change `www.dropbox.com` to `dl.dropboxusercontent.com` and remove `?dl=0`
4. Use `requests` to download:

```python
import requests

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        url = 'https://dl.dropboxusercontent.com/YOUR_FILE_PATH'
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device
```

---

## Using Git LFS for Large Files

If you want to keep the model in your repository:

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add the gitattributes file
git add .gitattributes

# Add your model
git add best_model.pth

# Commit and push
git commit -m "Add model file"
git push
```

---

## Environment Variables (Optional)

For sensitive information like API keys:

1. In Streamlit Cloud, go to your app settings
2. Add secrets in the **Secrets** section:

```toml
# .streamlit/secrets.toml format
model_url = "https://your-model-url.com/model.pth"
api_key = "your-api-key"
```

3. Access in code:
```python
model_url = st.secrets["model_url"]
```

---

## Monitoring Your Deployed App

- **View logs**: Click on "Manage app" → "Logs"
- **Restart app**: Click on "Manage app" → "Reboot"
- **Update app**: Push changes to GitHub, app auto-updates
- **Analytics**: Check viewer statistics in app settings

---

## Troubleshooting Deployment

### "App failed to load"
- Check logs for errors
- Verify all dependencies are in `requirements.txt`
- Check Python version compatibility

### "Out of resources"
- Streamlit Cloud free tier has resource limits
- Reduce memory usage
- Consider upgrading or using alternative hosting

### "ModuleNotFoundError"
- Ensure all imports are in `requirements.txt`
- Check for typos in package names

### Slow loading
- Model is being downloaded on first run
- Use `@st.cache_resource` decorator
- Consider smaller model or optimization

---

## Performance Tips

1. **Cache everything possible**:
```python
@st.cache_resource
def load_model():
    # Cached across all users
    pass

@st.cache_data
def load_data():
    # Cached per user session
    pass
```

2. **Optimize image size**:
```python
# Resize large images before processing
max_size = (1024, 1024)
image.thumbnail(max_size, Image.LANCZOS)
```

3. **Use session state**:
```python
if 'model' not in st.session_state:
    st.session_state.model = load_model()
```

---

## Custom Domain (Optional)

To use your own domain:

1. In Streamlit Cloud settings, add your custom domain
2. Update your DNS records:
   - CNAME: `your-app.streamlit.app`
3. Wait for DNS propagation

---

## Cost Considerations

**Streamlit Cloud Free Tier:**
- 1 private app
- Unlimited public apps
- Limited resources (1 CPU, 800MB RAM)
- Community support

**For production apps**, consider:
- Streamlit Cloud paid plans
- Self-hosting on AWS/GCP/Azure
- Heroku or Railway.app

---

## Next Steps

1. Test locally: `streamlit run app.py`
2. Push to GitHub
3. Deploy on Streamlit Cloud
4. Share your app URL!
5. Gather feedback and iterate

---

**Need Help?**
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Duality AI Discord](https://discord.com/invite/dualityfalconcommunity)

---

**🎉 Your app will be live and accessible to anyone with the link!**
