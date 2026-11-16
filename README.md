# TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video

**NeurIPS 2025 (Datasets & Benchmarks Track)**

[Finlay G. C. Hudson](https://finlay-hudson.github.io/), [James A. D. Gardner](https://jadgardner.github.io/), [William A. P. Smith](https://www-users.york.ac.uk/~waps101/) — University of York

[[Paper (arXiv:2511.21946)](https://arxiv.org/abs/2511.21946)] [[Project page](https://finlay-hudson.github.io/tapvid360/)] [[TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)] [[CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360)]

Given a narrow field-of-view video and query points in the first frame, TAPVid-360 asks for the 3D direction to each point in every frame, even after it leaves the camera's view. This repository contains the TAPVid360-10k evaluation code, the CoTracker360 baseline, and the code used to generate the dataset and train CoTracker360.


## Setup
Create a virtual environment (Python 3.10 or newer) and install the dependencies:

```
pip install -r requirements.txt
```

### If you wish to use the cotracker360 model 
```
pip install -r requirements-cotracker360.txt
```

## [Data](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)
To download the TAPVid360-10k dataset into `data/`, use the following commands:
```
mkdir -p data && cd data
git lfs install
git clone https://huggingface.co/datasets/fhudson96/TAPVid360-10k
cd TAPVid360-10k
tar -xvf dataset.tar.gz
```

## Evaluation

### Configs
Config file is located in `src/tapvid360/evaluation/eval_configs.py`

Put data in `data/` or change `data_dir` in config file

### If running through CoTracker360
Download pretrained model from [here](https://huggingface.co/fhudson96/cotracker360/blob/main/cotracker360.pth)

Put checkpoint in `checkpoints/` or change `trained_model_path` in config file

```
PYTHONPATH=. python src/tapvid360/evaluation/model_runner.py 
```

### If running through a separate model 
Outputs format should be a separate `.pth` file for each video in the dataset. With the `/` changed to a `-`. I.e `-0r64Y3PCTQ_clip_0000-0.pth`

Within this file should be a single tensor of shape `(N_frames, N_points, 3)` representing the unit vector (vc) coordinates of each point in each frame. For TAPVid360-10k this should be then of shape `(32, 256, 3)`

These should be stored under `results/` or change `results_dir` in config file
### Run TAPVid360-10k Evaluation
Within `src/tapvid360/evaluation/eval_runner.py` there is a flag called `RUN_METRICS` defaulted to `True`, if metrics have been generated this can be set to `False` to just aggregate the metrics.

```
PYTHONPATH=. python src/tapvid360/evaluation/eval_runner.py 
```


## Data Generation

The pipeline that built TAPVid360-10k from [360-1M](https://github.com/MattWallingford/360-1M) YouTube videos. It runs in three stages, plus a manual review step, all in `src/tapvid360/data_generation/`. Install the extra dependencies first, plus `ffmpeg` / `ffprobe` and, for the review tool, Tk (`sudo apt install ffmpeg python3-tk` on Ubuntu):

```
pip install -r requirements-cotracker360.txt -r requirements-generation.txt
```

Grounding DINO weights (via Hugging Face `transformers`), the LAION watermark detector and CoTracker3 (via `torch.hub`) download automatically. SAM 2 needs its Hiera-L checkpoint:

```
mkdir -p checkpoints/sam2
wget -P checkpoints/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 1. Download and filter clips (`download/`)
Downloads 360-1M videos (with at least 15 likes) and filters them as described in the paper. Coarse checks run on the full video: equirectangular metadata, borders, upside-down video, a visible seam, near-duplicate frames (LPIPS) and too little motion. The video is then split into 10-second clips, and each clip is checked for low optical flow, scene cuts and watermarks.

```
wget -P data https://huggingface.co/datasets/mwallingford/360-1M/resolve/main/360-1M.parquet
PYTHONPATH=. python src/tapvid360/data_generation/download/main.py \
    --metadata data/360-1M.parquet --out_dir outputs/download
```

The metadata file is [`360-1M.parquet`](https://huggingface.co/datasets/mwallingford/360-1M/resolve/main/360-1M.parquet) from the [360-1M dataset](https://huggingface.co/datasets/mwallingford/360-1M) (about 1.1M video IDs; only the `id` and `like_count` columns are used).

Clips are written to `outputs/download/<run>/filtered_clips/<video_id>/<video_id>_clip_NNNN.mp4`. Runs can be resumed with `--resume_dir outputs/download/<run>`. Downloads go through `yt-dlp`; pass `--cookies` if YouTube rate-limits you.

### 2. Track objects across the sphere (`tracks/`)
For 32 frames of each clip (resized to 2048×1024), Lang-SAM finds candidate moving objects and SAM 2 propagates their masks. CoTracker3 then tracks 256 points per object in an object-centred perspective video, and the tracks are projected back to the sphere. Static, poorly segmented and duplicate objects are dropped; thresholds are in `tracks/config.py`.

```
PYTHONPATH=. python src/tapvid360/data_generation/tracks/main.py \
    --clips_dir outputs/download/<run>/filtered_clips --out_dir outputs/tracks_360
```

When generating training data, pass `--exclude_list data/TAPVid360-10k` (the dataset root, or a JSON list of sample names such as `"-0r64Y3PCTQ_clip_0000/0"`) so that no TAPVid360-10k clip is reused. As for the released model, only those clips are excluded; other clips of the same videos can still be used. Stage 2 takes about 15 s per clip on an NVIDIA A40.

### 3. Manual review (`review/`)
Every object track in TAPVid360-10k was checked by hand. This tool plays each track on the equirectangular video or in an object-centred view. Press `A` to accept a track or `R` to reject it; decisions are saved to `outputs/tracks_360_manual_checking/`.

```
PYTHONPATH=. python src/tapvid360/data_generation/review/main.py --tracks_dir outputs/tracks_360
```

### 4. Resample camera trajectories and export (`sequences/`)
Each accepted object track is re-rendered as a 320×240 (75.18° horizontal FoV) video. The camera follows a simulated trajectory that starts centred on the object and can then leave it. The methods are spin (x/y/z axis), spiral, simulated hand-held camera (`PERSPECTIVE_SAMPLER`), random, re-identification (`REID`: move away and come back) and static.

```
# Benchmark (TAPVid360-10k): accepted tracks only, one random camera method per track
PYTHONPATH=. python src/tapvid360/data_generation/sequences/main.py \
    --tracks_dirs outputs/tracks_360 --out_dir outputs/TAPVid360/dataset --mode benchmark

# Training data: no review needed, 5 different camera methods per track
PYTHONPATH=. python src/tapvid360/data_generation/sequences/main.py \
    --tracks_dirs outputs/tracks_360_train --out_dir outputs/TAPVid360-training/dataset --mode training
```

Samples are written in the same format as the released dataset: `<clip>/<idx>/perspective_frames/*.jpg`, `data.pt` and `gen_info.txt`.

The camera methods in the released TAPVid360-10k are: RANDOM 4165, SPIN_Z 1546, SPIN_X 1469, SPIRAL 1386, PERSPECTIVE_SAMPLER 848, REID 567, SPIN_Y 13 and STATIC 6.

**Reproducibility:** a new run will not reproduce TAPVid360-10k exactly. YouTube videos get removed or changed, model outputs differ slightly between GPUs and library versions, the manual review is subjective, and the camera trajectories are random.

## CoTracker360 Training

CoTracker360 is fine-tuned from the CoTracker3 offline model. Its final layer is replaced so that it predicts a rotation for each point and frame, and the resulting directions are trained with a Huber loss. Generate training data with stages 1–4 above (with `--mode training`), then download the CoTracker3 weights and train:

```
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

PYTHONPATH=. python src/tapvid360/training/train.py \
    --train_dir outputs/TAPVid360-training --val_dir data/TAPVid360-10k \
    --cotracker3_checkpoint checkpoints/scaled_offline.pth --out_dir outputs/training/cotracker360
```

Defaults: 120 epochs, Adam with lr 1e-4, batch size 6, 32 query points over 32 frames, and validation on TAPVid360-10k every 5 epochs. Use `--resume` to continue a run. Checkpoints load directly into the evaluation code (`trained_model_path` in `eval_configs.py`).

**Note on the released weights:** the paper describes training for 120 epochs. Since the paper, we have continued training: the [released CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360) were trained for around 375 epochs, with batch size 6 (the paper states 8). To train for the same length, pass `--epochs 375` (with `--resume` to extend an existing run).



## Definitions

### Mapper
- f_x: focal length in pixels (x-direction)
- f_y: focal length in pixels (y-direction)
- fov_x: The horizontal field of view in degrees.
- fov_y: The vertical field of view in degrees.

### Conversions
- cr: pixel coordinates in the equirectangular image of shape *, 2
- uv: normalised coordinates in an equirectangular image of shape *, 2, with u for horizontal and v for vertical position
- thetaphi: Tensor of shape (*, 2) representing spherical coordinates (theta, phi) in radians
- vw: unit vectors in world coordinate system of shape (*, 3)
- vc: unit vectors in camera coordinate system of shape *,3 (this is what the coords in the dataset are represented in)
- ij: perspective pixel coordinates of shape *, 2

## Citation

```bibtex
@inproceedings{hudson2025tapvid,
  title         = {{TAPVid-360}: Tracking Any Point in 360 from Narrow Field of View Video},
  author        = {Hudson, Finlay G. C. and Gardner, James A. D. and Smith, William A. P.},
  booktitle     = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year          = {2025},
  eprint        = {2511.21946},
  archivePrefix = {arXiv}
}
```

## License

The code is MIT — see [LICENSE](LICENSE). The [TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k) and the [CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360) (derived from CoTracker3) are CC BY-NC 4.0.
