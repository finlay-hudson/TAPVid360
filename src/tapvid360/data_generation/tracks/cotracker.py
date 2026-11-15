"""CoTracker3 (offline) wrapper used to track query points through object-centred perspective videos."""
from typing import Optional

import torch

# CoTracker commit used to generate TAPVid360-10k (also pinned in requirements-cotracker360.txt).
COTRACKER_HUB_REF = "facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d"


def scale_points(points: torch.Tensor, old_size, new_size) -> torch.Tensor:
    """Scale (x, y) points from `old_size` to `new_size`, both given as (first, second) pairs.

    Note: the result is truncated to whole pixels (`.to(int)`), as in the pipeline that built TAPVid360-10k.
    """
    old_width, old_height = old_size
    new_width, new_height = new_size
    points = points.float()
    points[..., 0] *= new_width / old_width
    points[..., 1] *= new_height / old_height
    return points.to(int)


class CoTracker:
    def __init__(self, device=torch.device("cpu"), grid_size: int = 10, interp_shape=(384, 512)):
        self.model = torch.hub.load(COTRACKER_HUB_REF, "cotracker3_offline", trust_repo=True).to(device).eval()
        self.grid_size = grid_size
        self.interp_shape = interp_shape  # (H, W) videos are resized to before tracking.

    def preprocess(self, video: torch.Tensor, query_points: Optional[torch.Tensor] = None, device=torch.device("cpu")):
        """Resize (B, T, H, W, 3) videos to `interp_shape` and convert (B, N, 2) query points to CoTracker format.

        Note: point rescaling passes (H, W) sizes where (W, H) are expected. For 240x320 crops resized to 384x512
        both scale factors are 1.6, so this is exact; kept as-is to match how TAPVid360-10k was generated.
        """
        if video.ndim == 4:
            video = video.unsqueeze(0)
        videos = video.permute(0, 1, 4, 2, 3).float().to(device)
        videos_resize = torch.stack([torch.nn.functional.interpolate(v, tuple(self.interp_shape), mode="bilinear")
                                     for v in videos])
        if query_points is not None:
            query_points = scale_points(query_points, [videos.shape[-2], videos.shape[-1]], self.interp_shape)
            if query_points.ndim == 2:
                query_points = query_points.unsqueeze(0)
            query_points = torch.cat([torch.zeros_like(query_points[..., 0:1]), query_points],
                                     dim=-1).float().to(videos_resize.device)
        return videos_resize, query_points

    @torch.no_grad()
    def run(self, video: torch.Tensor, query_points: torch.Tensor):
        """Returns (tracks (B, T, N, 2), visibility (B, T, N))."""
        return self.model(video, segm_mask=None, queries=query_points, grid_size=self.grid_size)
