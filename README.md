# TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video

**NeurIPS 2025 (Datasets & Benchmarks Track)**

[Finlay G. C. Hudson](https://finlay-hudson.github.io/), [James A. D. Gardner](https://jadgardner.github.io/), [William A. P. Smith](https://www-users.york.ac.uk/~waps101/) — University of York

[[Paper (arXiv:2511.21946)](https://arxiv.org/abs/2511.21946)] [[Project page](https://finlay-hudson.github.io/tapvid360/)] [[TAPVid360-10k dataset](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)] [[CoTracker360 weights](https://huggingface.co/fhudson96/cotracker360)]

Given a narrow field-of-view video and query points in the first frame, TAPVid-360 asks for the 3D direction to each point in every frame, even after it leaves the camera's view. This repository contains the TAPVid360-10k evaluation code and the CoTracker360 baseline.


## Setup
Create a virtual environment and install the dependencies:

```
pip install -r requirements.txt
```

### If you wish to use the cotracker360 model 
```
pip install -r requirements-cotracker360.txt
```

## [Data](https://huggingface.co/datasets/fhudson96/TAPVid360-10k)
To download the TAPVid360-10k dataset, use the following commands:
```
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
Coming soon...

## Cotracker360 Training
Coming soon...



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
