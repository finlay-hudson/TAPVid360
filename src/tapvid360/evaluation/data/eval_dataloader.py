from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.tapvid360.evaluation.data.data_utils import DataSample


class TAPVid360Dataset(Dataset):
    def __init__(self, ds_root, transform=None, debug_max_items=-1, num_frames=32, num_queries=256,
                 specific_videos_list=None):
        super().__init__()
        self.ds_root = Path(ds_root) / "dataset"
        self.transform = transform
        self.spherical_img_names = [f"{item.parent.name}/{item.name}" for sublist in
                                    [list(vid_name.glob("*")) for vid_name in self.ds_root.glob("*")] for item in
                                    sublist]
        if specific_videos_list is not None:
            self.spherical_img_names = [vid for vid in self.spherical_img_names if vid in specific_videos_list]
        if debug_max_items > 0:
            if debug_max_items == 0:
                raise ValueError("debug_max_items must be greater than 1")
            self.spherical_img_names = self.spherical_img_names[:debug_max_items]
        self.num_frames = num_frames
        self.num_queries = num_queries

    def _get_the_data(self, spherical_img_name):
        t_imgs = []
        persp_img_shapes = []
        for i in range(self.num_frames):
            persp_img = Image.open(self.ds_root / spherical_img_name / "perspective_frames" / f"{i}.jpg")
            persp_img_shapes.append(persp_img.size)
            t_imgs.append(self.transform(persp_img))
        t_imgs = torch.stack(t_imgs)

        data = torch.load(self.ds_root / spherical_img_name / "data.pt", weights_only=False)
        _, _, w, h = t_imgs.shape

        return DataSample(
            video=t_imgs[:self.num_frames],
            trajectory=data["coords"][:self.num_frames, :self.num_queries],
            seq_name=spherical_img_name,
            rotations=data["rot_matrices"][:self.num_frames, :self.num_queries],
            equirectangular_height=data["equi_h"],
            orig_image_shapes=persp_img_shapes,
            visibility=data["coords_visibility"][:self.num_frames, :self.num_queries],
            fov_x=data["fov_x"],
        )

    def __len__(self):
        return len(self.spherical_img_names)

    def __getitem__(self, idx):
        return self._get_the_data(self.spherical_img_names[idx])
