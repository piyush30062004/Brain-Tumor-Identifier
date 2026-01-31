# NeuroScan AI - Brain Tumor Detection App

A professional brain tumor detection system powered by deep learning.

## 🚀 Deployment Instructions

### Files Required for Deployment:

1. **brain_tumor_app.py** - Main application file
2. **requirements.txt** - Python dependencies
3. **packages.txt** - System dependencies (for OpenCV)
4. **brain_tumor_model.keras** - Your trained model file
5. **.streamlit/config.toml** - App configuration (optional)

### Steps to Deploy on Streamlit Cloud:

1. **Push to GitHub:**
   ```bash
   git init
   git add brain_tumor_app.py requirements.txt packages.txt brain_tumor_model.keras
   git add .streamlit/config.toml
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository
   - Set main file path: `brain_tumor_app.py`
   - Click "Deploy"

### Important Notes:

- Make sure `brain_tumor_model.keras` is included in your repository
- The model file should be in the same directory as `brain_tumor_app.py`
- If your model file is large (>100MB), you may need to use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  ```

### Troubleshooting:

**If you get "installer returned a non-zero exit code":**
- Make sure `requirements.txt` has correct package versions
- Use `opencv-python-headless` instead of `opencv-python`
- Ensure `packages.txt` exists for system dependencies

**If model file is too large:**
- Use Git LFS (Large File Storage)
- Or host model on Google Drive/Dropbox and download it in the app

**Memory issues:**
- TensorFlow 2.15.0 is optimized for Streamlit Cloud
- Consider using a lighter model if issues persist

## 📦 Local Development

```bash
pip install -r requirements.txt
streamlit run brain_tumor_app.py
```

## 🎯 Features

- Professional desktop-like UI
- Real-time MRI scan analysis
- Confidence visualization with interactive charts
- Session statistics tracking
- Support for grayscale and RGB images

## ⚠️ Disclaimer

This application is for educational and research purposes only. Always consult with qualified medical professionals for diagnosis and treatment.
