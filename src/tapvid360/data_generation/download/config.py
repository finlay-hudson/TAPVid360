"""Settings for stage 1: downloading 360-1M videos and filtering them into 10-second 360° clips.

All thresholds match the pipeline used to build TAPVid360-10k (TAPVid-360 paper, Sec. 3.1).
"""

# --- Input selection ---
MIN_LIKE_COUNT = 15  # Videos need more than this many likes (quality baseline).

# --- Concurrency ---
MAX_DOWNLOAD_WORKERS = 3   # Concurrent yt-dlp downloads.
MAX_PROCESS_WORKERS = 5    # Concurrent filtering workers (CPU and RAM heavy).
VIDEO_QUEUE_SIZE = 20      # Downloaded videos waiting to be filtered.
STORE_FAILED = True        # Keep rejected videos/clips for inspection instead of deleting them.

# --- Run state (resumable) ---
RUN_STATE_FILENAME = "dataset_run.parquet"
STATE_UNPROCESSED = "U"
STATE_FAILED_DOWNLOAD = "FD"
STATE_DOWNLOADED = "D"
STATE_PASSED_COARSE = "PC"
STATE_CLIPS_GENERATED = "CG"
STATE_FAILED_COARSE = "FC"
STATE_FAILED_FINE = "FF"
STATE_SUCCESS = "S"

FINAL_STATES = {STATE_SUCCESS, STATE_FAILED_DOWNLOAD, STATE_FAILED_COARSE, STATE_FAILED_FINE}
TRACKED_STATES = {STATE_UNPROCESSED, STATE_FAILED_DOWNLOAD, STATE_FAILED_COARSE, STATE_FAILED_FINE, STATE_SUCCESS}

# --- Output sub-directories (inside each run directory) ---
TEMP_DOWNLOAD_DIR = "temp_downloads"
PASSED_COARSE_DIR = "passed_coarse"
CLIPS_DIR = "clips_coarse"
FILTERED_CLIPS_DIR = "filtered_clips"  # Final output: filtered_clips/<video_id>/<video_id>_clip_NNNN.mp4
FAILED_COARSE_DIR = "failed_coarse"
FAILED_FINE_DIR = "failed_fine"

# --- Coarse filtering (whole video) ---
NUM_SAMPLED_FRAMES = 10            # Evenly spaced frames used by the coarse checks.
LPIPS_SIMILARITY_THRESHOLD = 0.1   # Left/right or top/bottom halves this similar -> not a usable 360° video.
STATIC_THRESHOLD = 2.0             # Mean absolute grey-level difference between sampled frames.
SEAM_STRIP_WIDTH = 10              # Pixels taken from each edge for the wrap-around seam check.
SEAM_SIMILARITY_THRESHOLD = 0.4    # Normalised cross-correlation below this -> visible seam.

# --- Fine filtering (per 10-second clip) ---
CLIP_DURATION_S = 10
OPTICAL_FLOW_SAMPLE_INTERVAL = 2
OPTICAL_FLOW_DOWNSAMPLE = 0.5
OPTICAL_FLOW_MOTION_THRESHOLD = 0.2   # Mean Lucas-Kanade displacement (pixels) a clip must exceed.
WATERMARK_NUM_FRAMES = 10
WATERMARK_THRESHOLD = 0.5             # Mean LAION watermark probability above which a clip is rejected.
