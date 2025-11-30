# Depth Anything V2 - Pro Video Processor

A professional, robust, and hardware-agnostic implementation of monocular depth estimation for video. Optimized for Google Colab but compatible with local CPU/GPU environments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VikalpRajKisku/Depth_Anything_V2-Colab/blob/main/depth_launcher.ipynb)

## ✨ Key Features

- **🚀 Hardware Agnostic**: Automatically detects and uses **GPU (CUDA)** with FP16 acceleration for speed, or safely falls back to **CPU** for compatibility.
- **📏 Dual Modes**:
  - **Relative Depth**: Best for artistic visuals and creative effects.
  - **Metric Depth**: Estimates absolute distance (in meters) using specialized models.
- **🧊 3D Snapshots**: Export high-quality **3D Point Clouds (.ply)** from any specific frame in the video without generating massive files.
- **🛡️ Robust Engine**: Built-in memory management, flicker reduction (smoothing), and high-quality FFmpeg encoding.
- **⚡ Single-File Architecture**: All logic is contained within `depth_launcher.ipynb` - no external scripts required.

## ⚙️ Configuration Options

### Model Types

| Type       | Description                                                             |
| :--------- | :---------------------------------------------------------------------- |
| `Relative` | Standard visual depth map. Good for general video effects.              |
| `Metric`   | Absolute depth estimation. Good for measurements and 3D reconstruction. |

### Model Sizes (Relative Mode)

| Size    | Description                        |
| :------ | :--------------------------------- |
| `small` | Fastest, lower memory usage.       |
| `base`  | Balanced performance.              |
| `large` | Best quality, higher memory usage. |

### Resolutions

| Option   | Description                                  |
| :------- | :------------------------------------------- |
| `Native` | Original video resolution.                   |
| `720p`   | HD resolution.                               |
| `480p`   | Standard definition (Recommended for speed). |
| `360p`   | Fastest processing.                          |

## 🛠️ Installation (Local)

Since this is a single-notebook solution, you can simply run the notebook in Jupyter or VS Code.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/VikalpRajKisku/Depth_Anything_V2-Colab.git
    cd Depth_Anything_V2-Colab
    ```

2.  **Open `depth_launcher.ipynb`** in your preferred notebook environment.

3.  **Run the cells** sequentially. The notebook handles dependency installation automatically.

## 📂 Project Structure

| File                   | Description                                                      |
| :--------------------- | :--------------------------------------------------------------- |
| `depth_launcher.ipynb` | **The Core.** Contains the Engine, UI, and all processing logic. |

## 📦 Dependencies

The notebook automatically installs:

- `transformers`, `accelerate`, `torch` (Deep Learning)
- `opencv-python` (Video Processing)
- `yt-dlp` (Video Downloading)
- `pillow`, `numpy` (Image Ops)

## 🙏 Acknowledgments

This project uses the [Depth Anything V2](https://huggingface.co/depth-anything) model from Hugging Face Transformers.
