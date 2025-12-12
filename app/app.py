import os
import sqlite3
import tempfile
import shutil
import subprocess
from pathlib import Path
import streamlit as st
from PIL import Image
import time
from datetime import datetime

from face_recognition import FaceRecognitionEngine


# Page config
st.set_page_config(
    page_title="Face Recognition Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 1rem 0;
    }
    .timestamp {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: #667eea;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Directories
VIDEOS_DIR = "/data/videos"
QUERY_DIR = "/data/query"
WORK_DIR = "/data/output"

# Ensure directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(QUERY_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# Header
st.markdown('<h1 class="main-header">🎬 Face Recognition Demo</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    sample_rate = st.slider("Sample Rate (seconds)", 1, 5, 2, 
                          help="How often to extract frames from videos")
    similarity_threshold = st.slider("Similarity Threshold", 0.20, 0.60, 0.35, 0.05,
                                    help="Lower = stricter matching")
    min_face_score = st.slider("Min Face Detection Score", 0.40, 0.80, 0.60, 0.05,
                               help="Minimum confidence for face detection")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This demo uses:
    - **InsightFace** for face detection
    - **FAISS** for fast similarity search
    - **FFmpeg** for video processing
    
    Upload videos and a selfie to find where you appear!
    """)

# Check existing videos and selfie
existing_videos = [f for f in os.listdir(VIDEOS_DIR) 
                  if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))]
existing_selfie_path = os.path.join(QUERY_DIR, "selfie.jpg")
has_existing_selfie = os.path.exists(existing_selfie_path)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📹 Upload Videos")
    uploaded_videos = st.file_uploader(
        "Select video files",
        type=['mp4', 'mov', 'mkv', 'avi'],
        accept_multiple_files=True,
        help="Upload one or more video files"
    )
    
    # Show existing videos
    if existing_videos:
        st.info(f"📁 {len(existing_videos)} video(s) already in directory")
        for v in existing_videos[:5]:
            st.text(f"  • {v}")
        if len(existing_videos) > 5:
            st.caption(f"... and {len(existing_videos) - 5} more")
    
    if uploaded_videos:
        st.success(f"✅ {len(uploaded_videos)} new video(s) uploaded")
        for video in uploaded_videos:
            st.text(f"  • {video.name} ({video.size / 1024 / 1024:.1f} MB)")
            
            # Save video
            video_path = os.path.join(VIDEOS_DIR, video.name)
            with open(video_path, "wb") as f:
                f.write(video.getbuffer())
        
        st.rerun()

with col2:
    st.header("📸 Upload Selfie")
    
    if has_existing_selfie:
        st.info("📁 Selfie already in directory")
        try:
            existing_img = Image.open(existing_selfie_path)
            st.image(existing_img, caption="Current selfie")
        except:
            pass
    
    uploaded_selfie = st.file_uploader(
        "Select a selfie photo",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear front-facing photo of the person to find"
    )
    
    if uploaded_selfie:
        # Display selfie
        image = Image.open(uploaded_selfie)
        st.image(image, caption="Your selfie")
        
        # Save selfie
        selfie_path = os.path.join(QUERY_DIR, "selfie.jpg")
        with open(selfie_path, "wb") as f:
            f.write(uploaded_selfie.getbuffer())
        st.success("✅ Selfie uploaded")

# Process button
st.markdown("---")

# Check if we have videos (uploaded or existing)
has_videos = (uploaded_videos and len(uploaded_videos) > 0) or len(existing_videos) > 0

if st.button("🚀 Find My Face!", type="primary", use_container_width=True):
    if not has_videos:
        st.error("❌ Please upload at least one video")
    elif not uploaded_selfie and not has_existing_selfie:
        st.error("❌ Please upload a selfie")
    else:
        st.session_state.processing = True
        start_time = time.time()
        
        # Progress container
        progress_container = st.container()
        status_text = progress_container.empty()
        progress_bar = progress_container.progress(0)
        
        def update_progress(message):
            status_text.text(message)
            st.session_state.progress_message = message
        
        try:
            # Initialize engine
            status_text.text("🔧 Initializing face recognition engine...")
            progress_bar.progress(10)
            
            engine = FaceRecognitionEngine(
                videos_dir=VIDEOS_DIR,
                query_image=os.path.join(QUERY_DIR, "selfie.jpg"),
                work_dir=WORK_DIR,
                sample_every_seconds=sample_rate,
                similarity_threshold=similarity_threshold,
                min_face_det_score=min_face_score,
                progress_callback=update_progress
            )
            
            st.session_state.engine = engine
            
            # Build index
            progress_bar.progress(30)
            status_text.text("📼 Processing videos and extracting faces...")
            
            index, id_map = engine.build_index()
            
            progress_bar.progress(80)
            
            # Query
            status_text.text("🔍 Searching for matches...")
            results = engine.query_selfie(index, id_map)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.session_state.results = results
            st.session_state.processing = False
            st.session_state.processing_time = elapsed_time
            
            # Show results
            time.sleep(0.5)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False
            progress_bar.progress(0)

# Helper function to extract video clip
def extract_video_clip(video_path: str, timestamp: int, clip_duration: int = 8, output_dir: str = None) -> str:
    """Extract a short clip from video around the timestamp"""
    if output_dir is None:
        output_dir = os.path.join(WORK_DIR, "clips")
    os.makedirs(output_dir, exist_ok=True)
    
    # Start 3 seconds before, total duration 8 seconds
    start_time = max(0, timestamp - 3)
    clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{timestamp}s.mp4"
    output_path = os.path.join(output_dir, clip_name)
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(clip_duration),
        "-c", "copy",  # Copy codec for speed
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path if os.path.exists(output_path) else None
    except:
        return None

# Display results
if st.session_state.results:
    st.markdown("---")
    st.header("🎯 Results")
    
    # Show processing time
    if 'processing_time' in st.session_state:
        minutes = int(st.session_state.processing_time // 60)
        seconds = int(st.session_state.processing_time % 60)
        if minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        st.info(f"⏱️ **Processing time:** {time_str}")
    
    if not st.session_state.results:
        st.warning("No matches found. Try:")
        st.markdown("""
        - Lower the similarity threshold
        - Increase the sample rate (extract more frames)
        - Use a clearer selfie
        - Ensure good lighting in videos
        """)
    else:
        for video_name, timestamps in sorted(st.session_state.results.items()):
            with st.expander(f"🎬 {video_name} - {len(timestamps)} appearance(s)", expanded=True):
                video_path = os.path.join(VIDEOS_DIR, video_name)
                
                # Display timestamps
                st.markdown("**Timestamps:**")
                cols = st.columns(min(5, len(timestamps)))
                for i, ts in enumerate(timestamps[:20]):  # Show first 20
                    col_idx = i % 5
                    with cols[col_idx]:
                        h = ts // 3600
                        m = (ts % 3600) // 60
                        sec = ts % 60
                        st.markdown(f'<span class="timestamp">{h:02d}:{m:02d}:{sec:02d}</span>', 
                                  unsafe_allow_html=True)
                
                if len(timestamps) > 20:
                    st.caption(f"... and {len(timestamps) - 20} more")
                
                # Show video clips for first few matches
                st.markdown("---")
                st.markdown("**Video Clips:**")
                
                clips_to_show = min(5, len(timestamps))  # Show up to 5 clips
                for idx, ts in enumerate(timestamps[:clips_to_show]):
                    with st.container():
                        h = ts // 3600
                        m = (ts % 3600) // 60
                        sec = ts % 60
                        st.subheader(f"Clip {idx + 1}: {h:02d}:{m:02d}:{sec:02d}")
                        
                        if os.path.exists(video_path):
                            # Extract clip
                            clip_path = extract_video_clip(video_path, ts, clip_duration=8)
                            
                            if clip_path and os.path.exists(clip_path):
                                # Read video file and display
                                with open(clip_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            else:
                                st.warning("Could not extract clip. Showing frame instead.")
                                # Fallback to frame
                                try:
                                    engine = st.session_state.engine
                                    if engine:
                                        conn = sqlite3.connect(engine.db_path)
                                        cur = conn.cursor()
                                        cur.execute(
                                            "SELECT frame_path FROM faces WHERE video_name = ? AND ts_seconds = ? LIMIT 1",
                                            (video_name, ts)
                                        )
                                        row = cur.fetchone()
                                        conn.close()
                                        
                                        if row and os.path.exists(row[0]):
                                            frame_img = Image.open(row[0])
                                            st.image(frame_img, caption=f"Frame at {engine.seconds_to_hhmmss(ts)}")
                                except:
                                    pass
                        else:
                            st.error(f"Video file not found: {video_path}")
                        
                        if idx < clips_to_show - 1:
                            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with InsightFace, FAISS, and Streamlit</div>", 
          unsafe_allow_html=True)

