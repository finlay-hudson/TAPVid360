from tqdm import tqdm

import torch

from src.tapvid360.common.conversions.mapper import Mappers
from src.tapvid360.evaluation.data.data_utils import get_transform, collate_fn
from src.tapvid360.evaluation.data.eval_dataloader import TAPVid360Dataset
from src.tapvid360.evaluation.eval_configs import CoTracker360EvalConfig, TAPVid360EvalConfig
from src.tapvid360.evaluation.metrics.metrics_utils import gen_metrics_and_dump
from src.tapvid360.evaluation.model.cotracker360 import CoTracker360


def _make_query_points_for_cotracker(gt_unit_vectors, perspective_points):
    init_unit_vector = gt_unit_vectors[:, 0] if gt_unit_vectors.ndim == 4 else gt_unit_vectors
    init_persp_point = perspective_points[:, 0] if perspective_points.ndim == 4 else perspective_points

    return [torch.cat([torch.zeros_like(init_unit_vector[..., 0:1]), init_unit_vector], dim=-1), init_persp_point]


def _get_tapvid360_mapper(eq_hs, w, h, fov_xs):
    return [Mappers(w, h, equirectangular_height=eq_h, fov_x=75.18) for eq_h, fov_x in zip(eq_hs, fov_xs)]


@torch.no_grad()
def main(device, cotracker360_config, tapvid360_config):
    dataset = TAPVid360Dataset(tapvid360_config.data_dir, get_transform((240, 320)))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cotracker360_config.batch_size, shuffle=False,
                                             num_workers=cotracker360_config.num_workers, collate_fn=collate_fn)

    model = CoTracker360(stride=cotracker360_config.stride, corr_radius=cotracker360_config.corr_radius,
                         window_len=cotracker360_config.window_len,
                         model_resolution=cotracker360_config.model_resolution).to(device)
    model.load_weights(tapvid360_config.trained_model_path, strict=True)
    model.eval()

    tapvid360_config.results_dir.mkdir(parents=True, exist_ok=True)

    tapvid360_config.metrics_out_root.mkdir(exist_ok=True)
    metrics_out_path = tapvid360_config.metrics_out_root / tapvid360_config.data_dir.stem / "cotracker360"
    metrics_out_path.mkdir(exist_ok=True, parents=True)

    for i, sample in enumerate(tqdm(dataloader)):
        mappers = _get_tapvid360_mapper(sample.equirectangular_height, sample.video.shape[-1], sample.video.shape[-2],
                                        sample.fov_x)
        sample.to_device(device)
        query_points = torch.stack([mapper.point.vc.to_ij(sample.trajectory[b]) for b, mapper in enumerate(mappers)])
        query_points = _make_query_points_for_cotracker(sample.trajectory, query_points)

        coord_preds, vis_preds, confidence_preds, _ = model(sample.video, all_queries=query_points, mapper=mappers,
                                                            iters=cotracker360_config.num_iters)

        # dump results
        for b, seq_name in enumerate(sample.seq_name):
            save_seq_name = seq_name.replace("/", "-")  # avoid creating subfolders
            torch.save(coord_preds[b].cpu(), tapvid360_config.results_dir / f"{save_seq_name}.pth")
        gen_metrics_and_dump(coord_preds.cpu(), sample, metrics_out_path)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cotracker360_config = CoTracker360EvalConfig()
    tapvid360_config = TAPVid360EvalConfig()
    main(device, cotracker360_config, tapvid360_config)
