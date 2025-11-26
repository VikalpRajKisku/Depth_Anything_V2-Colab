Depth Anything V2 - Colab Video Processor

A professional implementation of monocular depth estimation for video, optimized for Google Colab stability and performance.

<!-- REPLACE 'YOUR_USERNAME' AND 'YOUR_REPO_NAME' IN THE LINK BELOW -->

🌟 Key Features

Temporal Smoothing: A rolling buffer averages depth maps across frames to remove jitter and flickering.

Smart Resolution: Downscale options (360p/480p) allow for significantly faster processing on T4 GPUs without breaking aspect ratios.

Professional Encoding: Output is encoded using FFmpeg with crf=18 (High Quality) and slow preset for optimal file size and compatibility.

Dual Input: Supports both File Uploads (drag-and-drop) and Direct URLs (YouTube/Direct Links).

🚀 How to Run (For Users)

Click the "Open in Colab" badge above.

Run the first cell.

The interface will appear automatically.

🛠️ Installation (Local Dev)

If you want to run this on your own machine:

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
python depth_engine.py --help

📂 Structure

depth_engine.py: The core logic class. Handles inference and video processing.

colab_notebook.py: The User Interface module imported by the launcher.

depth_launcher.ipynb: The entry point for Colab users.

requirements.txt: Python dependencies.
