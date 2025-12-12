# Local Face Recognition Demo

A fully local face recognition system that finds a person in videos using a selfie photo. No cloud services, no AWS bills.

## Features

✅ Processes **5 videos**  
✅ Takes a **selfie** as query  
✅ Outputs **which video(s)** + **timestamps** where the person appears  
✅ Runs entirely on your machine (zero cloud costs)

## Tech Stack

- **FFmpeg** - Frame extraction from videos
- **InsightFace** - Face detection + embeddings (via ONNX Runtime)
- **FAISS** - Fast similarity search
- **SQLite** - Stores timestamp ↔ frame ↔ video mapping
- **Streamlit** - Web GUI for easy demoing

## Setup

### 1. Add Your Videos

Place your 5 videos in the `videos/` directory:
- `v1.mp4`
- `v2.mp4`
- `v3.mp4`
- `v4.mp4`
- `v5.mp4`

Supported formats: `.mp4`, `.mov`, `.mkv`

### 2. Add Your Selfie

Place your selfie photo in the `query/` directory:
- `selfie.jpg`

**Tips for best results:**
- Front-facing photo
- Well-lit
- Clear face visibility
- Avoid sunglasses/obstructions

### 3. Build and Run

### Option A: Web GUI (Recommended for Demos)

From the project root:

```bash
docker build -t local-face-demo ./app
docker run --rm -p 8501:8501 \
  -v "$PWD/videos:/data/videos" \
  -v "$PWD/query:/data/query" \
  -v "$PWD/output:/data/output" \
  local-face-demo
```

Then open your browser to: **http://localhost:8501**

### GUI Features

The web interface provides:
- 📹 **Video Upload** - Drag and drop multiple videos
- 📸 **Selfie Upload** - Upload your query photo with preview
- ⚙️ **Live Configuration** - Adjust settings without rebuilding:
  - Sample rate (how often to extract frames)
  - Similarity threshold (matching strictness)
  - Face detection confidence
- 📊 **Real-time Progress** - See processing status as it happens
- 🎯 **Rich Results** - View matches with:
  - Video names and timestamps
  - Preview frames from matches
  - Organized by video

### Option B: Command Line

If you prefer the command-line interface, use the original `main.py`:

```bash
docker build -t local-face-demo ./app
docker run --rm -it \
  -v "$PWD/videos:/data/videos" \
  -v "$PWD/query:/data/query" \
  -v "$PWD/output:/data/output" \
  local-face-demo python main.py
```

## Output

The system will:
1. Extract frames from all videos (every 2 seconds by default)
2. Detect and embed all faces found
3. Build a searchable index
4. Query with your selfie
5. Return matches with timestamps

Example output:
```
✅ Matches (video → timestamps):
 - v2.mp4: 00:12:14, 00:12:18, 00:12:22
 - v5.mp4: 00:33:08
```

## Tuning Parameters

Edit `app/main.py` to adjust:

### If it misses matches:
- `SAMPLE_EVERY_SECONDS = 1` (more frames = better accuracy, slower)
- `SIMILARITY_THRESHOLD = 0.30` (looser matching)
- Ensure selfie is front-facing + well-lit

### If it returns false matches:
- `SIMILARITY_THRESHOLD = 0.45` or `0.55` (stricter matching)
- `MIN_FACE_DET_SCORE = 0.70` (stricter face detection)

**Rule of thumb:** Event footage is challenging; you'll likely need to tune the threshold based on your specific videos.

## Project Structure

```
local-face-demo/
  videos/          # Place your 5 videos here
    v1.mp4
    v2.mp4
    ...
  query/           # Place your selfie here
    selfie.jpg
  app/
    main.py        # Main processing script
    requirements.txt
    Dockerfile
  output/          # Generated frames, index, database
    frames/        # Extracted video frames
    faces.db       # SQLite database
    faces.index    # FAISS index
    id_map.json    # ID mapping
```

## How It Works

1. **Frame Extraction**: FFmpeg extracts frames every N seconds from each video
2. **Face Detection**: InsightFace detects all faces in each frame
3. **Embedding**: Each face is converted to a 512-dimensional vector
4. **Indexing**: All embeddings are stored in a FAISS index for fast similarity search
5. **Query**: Your selfie is embedded and searched against the index
6. **Results**: Matches are filtered by similarity threshold and timestamps are merged/compacted

## Notes

- The system processes all faces found in frames (works for crowds)
- Timestamps are automatically merged if they're within 4 seconds of each other
- The index is rebuilt each run (simple for demo purposes)
- All processing happens locally in the Docker container

