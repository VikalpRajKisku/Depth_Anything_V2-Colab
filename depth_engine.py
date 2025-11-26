import os
import sys
import logging
import subprocess
import shutil
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, Union

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import yt_dlp

# --- Professional Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DepthEngine")

class DepthVideoEngine:
    """
    Core logic for processing videos with Depth Anything V2.
    Handles model inference, temporal smoothing, and high-quality encoding.
    """

    def __init__(self, model_size: str = "small", device: str = None):
        """
        Initialize the engine.
        
        Args:
            model_size: 'small', 'base', or 'large'.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = f"depth-anything/Depth-Anything-V2-{model_size.title()}-hf"
        
        logger.info(f"Initializing Engine on {self.device.upper()}...")
        logger.info(f"Loading checkpoint: {self.checkpoint}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            raise RuntimeError("Model initialization failed.") from e

    def download_video(self, source_url: str) -> str:
        """
        Downloads video from a URL (Direct or YouTube).
        Returns the path to the downloaded file.
        """
        filename = f"input_{int(datetime.now().timestamp())}.mp4"
        logger.info(f"Acquiring video from: {source_url}")

        # 1. Try generic HTTP download first (faster for direct links)
        if source_url.lower().endswith(('.mp4', '.mov', '.avi')):
            import urllib.request
            try:
                urllib.request.urlretrieve(source_url, filename)
                logger.info("Direct HTTP download successful.")
                return filename
            except Exception as e:
                logger.warning(f"Direct download failed: {e}. Falling back to yt-dlp.")

        # 2. Use yt-dlp for everything else (YouTube, etc.)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': filename,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source_url])
            logger.info("yt-dlp download successful.")
            return filename
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def process_video(
        self, 
        input_path: str, 
        resolution_p: int = 480, 
        colormap: str = "gray", 
        smooth_window: int = 3
    ) -> Optional[str]:
        """
        Main processing pipeline.
        
        Args:
            input_path: Path to the local input video.
            resolution_p: Target height (e.g., 480, 720).
            colormap: 'gray', 'inferno', 'magma', 'jet', etc.
            smooth_window: Number of frames to average for flicker reduction.
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return None

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("Failed to open video capture.")
            return None

        # Input Stats
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate Output Dims
        target_w, target_h = self._get_dims(orig_w, orig_h, resolution_p)
        logger.info(f"Processing: {orig_w}x{orig_h} -> {target_w}x{target_h} @ {fps}fps")

        # Setup Intermediate Writer (MJPG is safe/lossless-ish for temp)
        temp_avi = "temp_processing.avi"
        out = cv2.VideoWriter(
            temp_avi, 
            cv2.VideoWriter_fourcc(*'MJPG'), 
            fps, 
            (target_w, target_h)
        )

        # Smoothing Buffer
        buffer = deque(maxlen=smooth_window)
        
        # Colormap Enum
        if colormap.lower() == "gray":
            cmap_enum = None
        else:
            cmap_enum = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)

        logger.info("Starting inference loop...")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break

                # 1. Resize Input (Performance Optimization)
                if (target_w, target_h) != (orig_w, orig_h):
                    frame_in = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    frame_in = frame

                # 2. Inference
                inputs = self.processor(
                    images=cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB), 
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    depth = self.model(**inputs).predicted_depth

                # 3. High-Quality Upscaling (Bicubic)
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze().cpu().numpy()

                # 4. Temporal Smoothing
                buffer.append(depth)
                avg_depth = np.mean(buffer, axis=0)

                # 5. Normalization
                d_min, d_max = avg_depth.min(), avg_depth.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (avg_depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(avg_depth)
                
                depth_uint8 = (depth_norm * 255).astype(np.uint8)

                # 6. Colorize
                if cmap_enum is None:
                    final = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
                else:
                    final = cv2.applyColorMap(depth_uint8, cmap_enum)

                out.write(final)

                frame_idx += 1
                if frame_idx % 50 == 0:
                    logger.info(f"Progress: {frame_idx}/{total_frames} frames")

        except KeyboardInterrupt:
            logger.warning("User interrupted processing.")
        finally:
            cap.release()
            out.release()

        # Final Encode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"depth_out_{timestamp}_{resolution_p}p.mp4"
        self._encode_ffmpeg(temp_avi, output_name)
        
        return output_name

    def _get_dims(self, w, h, target_p):
        if target_p >= h: return w, h
        scale = target_p / h
        new_w = int(w * scale)
        if new_w % 2 != 0: new_w -= 1 # Even width required for h264
        return new_w, target_p

    def _encode_ffmpeg(self, input_path, output_path):
        logger.info("Encoding final H.264 video (High Quality)...")
        if os.path.exists(output_path): os.remove(output_path)
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p", # Essential for compatibility
            "-crf", "18",          # Visually Lossless
            "-preset", "slow",     # Better compression efficiency
            "-loglevel", "error",
            output_path
        ]
        subprocess.run(cmd, check=True)
        if os.path.exists(input_path): os.remove(input_path)
        logger.info(f"Saved: {output_path}")