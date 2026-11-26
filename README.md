# TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video

**NeurIPS 2025 (Datasets & Benchmarks Track)**

[Finlay G. C. Hudson](https://finlay-hudson.github.io/), [James A. D. Gardner](https://jadgardner.github.io/), [William A. P. Smith](https://www-users.york.ac.uk/~waps101/) — University of York

[[Paper (arXiv:2511.21946)](https://arxiv.org/abs/2511.21946)] [[Project page](https://finlay-hudson.github.io/tapvid360/)] [[TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)] [[CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360)]

Given a narrow field-of-view video and query points in the first frame, TAPVid-360 asks for the 3D direction to each point in every frame, even after it leaves the camera's view. This repository contains the TAPVid360-10k evaluation code, the CoTracker360 baseline, and the code used to generate the dataset and train CoTracker360.

## News
**26 Nov 2026: in-frame labels corrected in TAPVid360-10k v1.1.** We found a bug in the in-frame / out-of-frame labels of TAPVid360-10k v1.0, the version used in the paper. Points behind the camera could be labelled in frame. This affected 10.8% of the in-frame labels. v1.1 fixes the labels and changes nothing else in the dataset. Results over all points are unchanged. With the corrected labels, every method scores better in frame and worse out of frame. CoTracker360 is still clearly the best method out of frame. See [Dataset versions](#dataset-versions) for the cause and the corrected Tables 2–4. v1.0 remains available.

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
cd ../..
```

Without git-lfs (e.g. on a shared cluster), download with the Hugging Face CLI instead:
```
pip install -U huggingface_hub
mkdir -p data && cd data
hf download fhudson96/TAPVid360-10k --repo-type dataset --local-dir TAPVid360-10k
cd TAPVid360-10k
tar -xvf dataset.tar.gz
cd ../..
```

### Dataset versions
- **v1.1 (current):** corrected in-frame labels (`coords_visibility`).
- **v1.0:** the version used in the paper. Download it with `--revision v1.0` (`hf download`) or `git checkout v1.0` (git clone).

In v1.0, points *behind* the camera could be labelled in frame, because their projection to pixels can also land inside the image. This affected 3.94M (10.8%) of the point labels marked in frame, mostly in the `SPIN_X` and `SPIN_Z` camera methods. v1.1 changes only `coords_visibility`. The videos, ground-truth directions (`coords`) and rotations are identical, so results over all points are unchanged.

To update a v1.0 copy yourself, apply this to each sample's `data.pt`:
```python
d["coords_visibility"] = d["coords_visibility"] & (d["coords"][..., 0] > 0)
```

#### What changed from v1.0 to v1.1
In the tables below, IF means in frame and OOF out of frame. Each cell gives the v1.0 → v1.1 value. A single value means the metric did not change. The "all" columns never change, because they don't use the labels.

**Table 2 (points):** in frame 36.28M → 32.34M, out of frame 45.64M → 49.58M.

**Table 3:**

| Method | < δ all ↑ | < δ IF ↑ | < δ OOF ↑ | AD all ↓ | AD IF ↓ | AD OOF ↓ |
|---|---|---|---|---|---|---|
| TAPNext | 0.0083 | 0.0194 → 0.0219 | 0.0005 → 0.0004 | 51.75 | 36.49 → 23.44 | 62.14 → 66.78 |
| TAPIR | 0.0106 | 0.0251 → 0.0282 | 0.0003 | 49.81 | 33.88 → 19.90 | 60.52 → 65.82 |
| BootsTAPIR | 0.0120 | 0.0280 → 0.0314 | 0.0005 | 48.09 | 33.20 → 19.55 | 57.99 → 63.27 |
| TAPIP3D* | 0.2477 | 0.4726 → 0.5335 | 0.0831 → 0.0820 | 36.21 | 23.01 → 10.82 | 45.60 → 49.38 |
| SpatialTracker | 0.2239 | 0.4893 → 0.5583 | 0.0303 → 0.0295 | 38.88 | 22.16 → 8.34 | 50.43 → 54.72 |
| CoTracker3 (offline)* | 0.2405 | 0.5546 → 0.6291 | 0.0152 → 0.0144 | 37.71 | 17.81 → 3.53 | 51.29 → 55.82 |
| CoTracker360 | 0.2386 | 0.4060 → 0.4616 | 0.1160 → 0.1123 | 8.27 | 3.95 → 2.93 | 10.98 → 11.17 |

**Table 4 (CoTracker360 by camera method):** in the paper, `REID` is called *btf* and `PERSPECTIVE_SAMPLER` *simulated human*.

| Camera method | Samples | < δ IF ↑ | < δ OOF ↑ | AD IF ↓ | AD OOF ↓ |
|---|---|---|---|---|---|
| `SPIN_X` | 1469 | 0.4537 → 0.6266 | 0.1255 → 0.1158 | 3.92 → 1.55 | 7.84 → 8.12 |
| `SPIN_Z` | 1546 | 0.4255 → 0.5998 | 0.1379 → 0.1244 | 4.40 → 1.85 | 7.62 → 8.08 |
| `PERSPECTIVE_SAMPLER` | 848 | 0.4441 → 0.4762 | 0.0849 → 0.0822 | 3.90 → 2.26 | 12.20 → 12.84 |
| `RANDOM` | 4165 | 0.3300 → 0.3311 | 0.0671 | 4.39 → 4.09 | 14.40 → 14.44 |
| `REID` | 567 | 0.4128 → 0.4129 | 0.0568 | 4.09 → 4.07 | 15.13 → 15.14 |
| `SPIRAL` | 1386 | 0.5367 → 0.5373 | 0.2733 | 2.09 → 1.98 | 4.94 → 4.96 |
| `SPIN_Y` | 13 | 0.3412 | 0.0035 | 7.39 | 49.23 |
| `STATIC` | 6 | 0.3212 | 0.0207 | 5.13 | 24.10 |

The v1.0 values were rescored with this repository's evaluation code, using the paper's saved predictions. The TAPIR, BootsTAPIR and TAPNext predictions had not been kept, so those three were re-run. These v1.0 values match the paper's Table 3 to within 0.003 accuracy and 0.3°. CoTracker360 matches Tables 3 and 4 exactly. \*TAPIP3D was scored over 1,279 samples and CoTracker3 over 8,137, the samples with saved predictions.

The paper's conclusions do not change: CoTracker360 is still clearly the best method out of frame. In frame, the baselines' errors are much smaller than v1.0 suggested. For example, CoTracker3's in-frame AD drops from 17.81° to 3.53°.

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

With the released weights and dataset v1.0, this reproduces the CoTracker360 results in Tables 3 and 4 of the paper (e.g. 0.2386 accuracy and 8.27° mean angular error over all points), up to small floating-point differences between GPUs and library versions. With v1.1, the in-frame and out-of-frame results differ (see [Dataset versions](#dataset-versions)).

### If running through a separate model 
Outputs format should be a separate `.pth` file for each video in the dataset. With the `/` changed to a `-`. I.e `-0r64Y3PCTQ_clip_0000-0.pth`

Within this file should be a single tensor of shape `(N_frames, N_points, 3)` representing the unit vector (vc) coordinates of each point in each frame. For TAPVid360-10k this should be then of shape `(32, 256, 3)`

These should be stored under `results/` or change `results_dir` in config file, and `method_name` in the config file set to the name of the sub-folder of `results/` holding them.

The scripts for the other baselines in the paper (TAPIR, BootsTAPIR, TAPNext, TAPIP3D, SpatialTracker, CoTracker3) are not included. As described in the paper, their pixel tracks were converted to directions using the camera intrinsics, which `Mappers` does:

```python
from src.tapvid360.common.conversions.mapper import Mappers

# tracks: (32, 256, 2) pixel (x, y) predictions in the 320x240 frames; data = torch.load(<sample>/data.pt)
mapper = Mappers(320, 240, equirectangular_height=data["equi_h"], fov_x=data["fov_x"])
directions = mapper.point.ij.to_vc(tracks)  # (32, 256, 3) unit vectors, the format described above
```

### Run TAPVid360-10k Evaluation
Within `src/tapvid360/evaluation/eval_runner.py` there is a flag called `RUN_METRICS` defaulted to `True`, if metrics have been generated this can be set to `False` to just aggregate the metrics. Metrics are reported over the whole dataset and per camera method (as in Table 4 of the paper).

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

Clips are written to `outputs/download/<run>/filtered_clips/<video_id>/<video_id>_clip_NNNN.mp4`. Most videos are rejected, so expect many `failed coarse filtering: ...` log lines (with the reason) before the first clips appear; `successful` and `filtered_clips_total` in the `Progress` lines count what passed. Decoder messages such as `libav.h264 - ERROR - mmco: unref short failure` and torchvision's `pretrained` deprecation warnings are harmless. Runs can be resumed with `--resume_dir outputs/download/<run>`. Downloads go through `yt-dlp`; pass `--cookies` if YouTube rate-limits you.

### 2. Track objects across the sphere (`tracks/`)
For 32 frames of each clip (resized to 2048×1024), Lang-SAM finds candidate moving objects and SAM 2 propagates their masks. CoTracker3 then tracks 256 points per object in an object-centred perspective video, and the tracks are projected back to the sphere. Static, poorly segmented and duplicate objects are dropped; thresholds are in `tracks/config.py`.

```
PYTHONPATH=. python src/tapvid360/data_generation/tracks/main.py \
    --clips_dir outputs/download/<run>/filtered_clips --out_dir outputs/tracks_360
```

When generating training data, pass `--exclude_list data/TAPVid360-10k` (the dataset root, or a JSON list of sample names such as `"-0r64Y3PCTQ_clip_0000/0"`) so that no TAPVid360-10k clip is reused. As for the released model, only those clips are excluded; other clips of the same videos can still be used. Stage 2 takes about 15 s per clip on an NVIDIA A40. Reruns skip finished clips, including those listed in `clips_without_tracks.txt` (no object survived the filters).

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

**Names in the paper:** `REID` is *btf* (back-to-front) and `PERSPECTIVE_SAMPLER` is *simulated human*; the other methods have the same names.

**Where the code differs from the paper's description:** the code is what generated TAPVid360-10k, so it is released as is.
- Motion filter (stage 2): objects are kept or dropped as a whole, by the mean path length of their points, rather than each point's track being filtered by its own path length.
- Scene-dynamics check (stage 1): the mean absolute difference between evenly spaced frames, rather than pixel variance at random intervals.
- Top/bottom LPIPS check (stage 1): a match rejects the video; only videos whose metadata says top-and-bottom are cropped.
- The second-stage Viser checking tool (Appendix A, Fig. 7) is not included.

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

The code is MIT — see [LICENSE](LICENSE) — except `src/tapvid360/common/model/cotracker360.py`, which is modified from [CoTracker](https://github.com/facebookresearch/co-tracker) and so is CC BY-NC 4.0. Parts of `src/tapvid360/data_generation/tracks/` are adapted from [SAM 2](https://github.com/facebookresearch/sam2) and [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything) (both Apache-2.0). The [TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k) and the [CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360) (derived from CoTracker3) are CC BY-NC 4.0.
