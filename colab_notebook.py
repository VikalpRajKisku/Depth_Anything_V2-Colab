# @title 🖥️ Depth Anything V2 - Dashboard
# @markdown Select your settings and run the cell.

import os
import sys
from google.colab import files
from IPython.display import Video, display, clear_output

# 1. Dependency Check
if not os.path.exists("depth_engine.py"):
    print("⚠️ 'depth_engine.py' not found. Please upload the file or clone the repo.")
else:
    # Install libs if not present (Quietly)
    try:
        import yt_dlp
    except ImportError:
        print("⚙️ Installing Dependencies...")
        !pip install -q git+https://github.com/huggingface/transformers.git accelerate opencv-python yt-dlp torch pillow

# Import the engine from the other file
from depth_engine import DepthVideoEngine

# --- UI Configuration ---
SOURCE_TYPE = "URL" # @param ["URL", "Upload"]
VIDEO_URL = "https://media.w3.org/2010/05/sintel/trailer.mp4" # @param {type:"string"}
MODEL_SIZE = "small" # @param ["small", "base", "large"]
RESOLUTION = "480p" # @param ["Native", "720p", "480p", "360p"]
COLORMAP = "gray" # @param ["gray", "inferno", "magma", "jet", "turbo"]
SMOOTHING = 3 # @param {type:"slider", min:1, max:10, step:1}

def run_dashboard():
    # 1. Input Handling
    input_path = None
    if SOURCE_TYPE == "Upload":
        print("\n📂 Please upload your video:")
        uploaded = files.upload()
        if not uploaded:
            print("❌ No file uploaded.")
            return
        input_path = list(uploaded.keys())[0]
    else:
        input_path = VIDEO_URL

    # 2. Input Preview (If local)
    if os.path.exists(str(input_path)):
        print("\n🎬 Input Preview:")
        display(Video(input_path, embed=True, width=400))
    
    # 3. Initialize Engine
    print("\n🔧 Initializing Engine...")
    engine = DepthVideoEngine(model_size=MODEL_SIZE)

    # 4. Download (if URL) & Process
    try:
        final_video = None
        
        # If URL, download via engine logic first
        if SOURCE_TYPE == "URL":
            input_path = engine.download_video(VIDEO_URL)
            print("\n🎬 Input Preview:")
            display(Video(input_path, embed=True, width=400))
            
        # Parse resolution
        res_p = int(RESOLUTION.replace("p", "")) if RESOLUTION != "Native" else 2160
        if RESOLUTION == "Native": res_p = 4320 

        # --- THE MAIN CALL ---
        final_video = engine.process_video(
            input_path=input_path,
            resolution_p=res_p,
            colormap=COLORMAP,
            smooth_window=SMOOTHING
        )

        # 5. Output Preview & Download
        if final_video and os.path.exists(final_video):
            print("\n✨ Processing Complete!")
            print(f"📄 Output: {final_video}")
            display(Video(final_video, embed=True, width=400))
            
            print("⬇️ Triggering Download...")
            files.download(final_video)
        else:
            print("❌ Processing failed.")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == "__main__":
    run_dashboard()