# Depth Anything V2 - Colab Video Processor

A professional implementation of monocular depth estimation for video, optimized for Google Colab stability and performance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VikalpRajKisku/Depth_Anything_V2-Colab/blob/main/depth_launcher.ipynb)

## 🌟 Key Features

- **Temporal Smoothing:** A rolling buffer averages depth maps across frames to remove jitter and flickering.
- **Smart Resolution:** Downscale options (360p/480p) allow for significantly faster processing on T4 GPUs without breaking aspect ratios.
- **Professional Encoding:** Output is encoded using FFmpeg with `crf=18` (High Quality) and `slow` preset for optimal file size and compatibility.
- **Dual Input:** Supports both File Uploads (drag-and-drop) and Direct URLs (YouTube/Direct Links).

## 🚀 How to Run (For Users)

1. Click the "Open in Colab" badge above.
2. Run the first cell.
3. The interface will appear automatically.

## ⚙️ Configuration Options

### Model Sizes

| Size | Description |
|------|-------------|
| `small` | Fastest, lower memory usage |
| `base` | Balanced performance |
| `large` | Best quality, higher memory usage |

### Resolutions

| Option | Description |
|--------|-------------|
| `Native` | Original video resolution |
| `720p` | HD resolution |
| `480p` | Standard definition (recommended for T4 GPUs) |
| `360p` | Fastest processing |

### Colormaps

| Option | Description |
|--------|-------------|
| `gray` | Grayscale depth map |
| `inferno` | Warm color gradient |
| `magma` | Purple to yellow gradient |
| `jet` | Rainbow color gradient |
| `turbo` | Improved rainbow gradient |

## 🛠️ Installation (Local Dev)

If you want to run this on your own machine:

```bash
git clone https://github.com/VikalpRajKisku/Depth_Anything_V2-Colab.git
cd Depth_Anything_V2-Colab
pip install -r requirements.txt
python depth_engine.py --help
```

## 📂 Project Structure

| File | Description |
|------|-------------|
| `depth_engine.py` | The core logic class. Handles inference and video processing. |
| `colab_notebook.py` | The User Interface module imported by the launcher. |
| `depth_launcher.ipynb` | The entry point for Colab users. |
| `requirements.txt` | Python dependencies. |

## 📦 Dependencies

- `torch` & `torchvision` - Deep learning framework
- `transformers` - Hugging Face model loading
- `accelerate` - Optimized inference
- `opencv-python` - Video processing
- `yt-dlp` - YouTube video downloading
- `pillow` - Image processing
- `numpy` - Numerical computations

## 🙏 Acknowledgments

This project uses the [Depth Anything V2](https://huggingface.co/depth-anything) model from Hugging Face Transformers.
