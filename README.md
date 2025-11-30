# 🚀 Depth Anything V2 - Pro Video Processor

**Robust. Hardware-Agnostic. Advanced.**

A professional implementation of monocular depth estimation for video, optimized for Google Colab stability and performance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VikalpRajKisku/Depth_Anything_V2-Colab/blob/main/depth_launcher.ipynb)

## ✨ Features

- **Dual Modes**: Standard Relative Depth (Visuals) & Metric Depth (Measurements)
- **Hardware Smart**: Automatically uses GPU (FP16) for speed or CPU (FP32) for compatibility
- **3D Snapshots**: Export high-quality 3D Point Clouds (.ply) from any frame
- **Robust Engine**: Flicker reduction, high-quality FFmpeg encoding, and memory safety
- **Flexible Input**: Upload videos directly or download from YouTube/URLs

## 🎯 Quick Start

1. Click the "Open in Colab" badge above
2. Run Cell 1 to install dependencies
3. Run Cell 2 to initialize the depth engine
4. Configure your settings in Cell 3 and run the dashboard

## ⚙️ Configuration Options

### Model Types

| Type       | Description                                      |
| ---------- | ------------------------------------------------ |
| `Relative` | Best for visual depth maps and artistic effects  |
| `Metric`   | Outputs actual depth measurements for 3D export  |

### Model Sizes (Relative Mode)

| Size    | Description                       |
| ------- | --------------------------------- |
| `small` | Fastest, lower memory usage       |
| `base`  | Balanced performance              |
| `large` | Best quality, higher memory usage |

### Output Resolutions

| Option   | Description                                   |
| -------- | --------------------------------------------- |
| `Native` | Original video resolution                     |
| `720p`   | HD resolution                                 |
| `480p`   | Standard definition (recommended for T4 GPUs) |
| `360p`   | Fastest processing                            |

## 🧊 3D Point Cloud Export

Generate 3D point clouds (.ply files) from any frame in your video:

1. Set `GENERATE_SNAPSHOT = True`
2. Set `SNAPSHOT_TIME` to the desired timestamp (in seconds)
3. The .ply file will be automatically downloaded after processing

Point clouds can be viewed in software like MeshLab, Blender, or CloudCompare.

## 📂 Project Structure

| File                   | Description                               |
| ---------------------- | ----------------------------------------- |
| `depth_launcher.ipynb` | Main Colab notebook with all-in-one setup |

## 📦 Dependencies

The notebook automatically installs all required packages (PyTorch is pre-installed in Colab):

- `transformers` (from source) - Hugging Face model loading
- `accelerate` - Optimized inference
- `opencv-python` - Video processing
- `yt-dlp` - YouTube/URL video downloading
- `pillow` - Image processing
- `numpy` - Numerical computations

## 💡 Tips

- Use `480p` resolution for T4 GPUs to avoid memory issues
- The `small` model provides good results with faster processing
- Enable temporal smoothing (default: 3 frames) for flicker-free output
- Metric mode is recommended when exporting 3D point clouds

## 🙏 Acknowledgments

This project uses the [Depth Anything V2](https://huggingface.co/depth-anything) model from Hugging Face Transformers.
