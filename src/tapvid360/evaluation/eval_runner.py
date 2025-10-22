from tqdm import tqdm

import torch

from src.tapvid360.evaluation.data.data_utils import get_transform, collate_fn
from src.tapvid360.evaluation.data.eval_dataloader import TAPVid360Dataset
from src.tapvid360.evaluation.eval_configs import TAPVid360EvalConfig, CoTracker360EvalConfig
from src.tapvid360.evaluation.metrics.metrics_utils import aggregate_metrics, gen_metrics_and_dump

RUN_METRICS = True


def _get_metrics(tapvid360_config, model_config):
    dataset = TAPVid360Dataset(tapvid360_config.data_dir, get_transform(model_config.model_resolution))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=model_config.batch_size, shuffle=False,
                                             num_workers=model_config.num_workers, collate_fn=collate_fn)

    tapvid360_config.metrics_out_root.mkdir(exist_ok=True)
    metrics_out_path = tapvid360_config.metrics_out_root / tapvid360_config.data_dir.stem / "cotracker360"
    metrics_out_path.mkdir(exist_ok=True, parents=True)

    for i, sample in enumerate(tqdm(dataloader)):
        res_path = tapvid360_config.results_dir / tapvid360_config.method_name / f"{sample.seq_name[0].replace('/', '-')}.pth"
        assert res_path.exists(), f"Results not found for {sample.seq_name[0]}"
        results = torch.load(res_path, weights_only=False, map_location="cpu")
        gen_metrics_and_dump(results, sample, metrics_out_path)


if __name__ == "__main__":
    cfg = TAPVid360EvalConfig()
    metrics_dir = cfg.metrics_out_root / cfg.data_dir.stem

    model_config = CoTracker360EvalConfig()
    tapvid360_config = TAPVid360EvalConfig()

    if RUN_METRICS:
        assert (tapvid360_config.results_dir / cfg.method_name).exists(), (f"Need to generate results for "
                                                                           f"{cfg.method_name}, instructions within "
                                                                           f"README.md")
        _get_metrics(tapvid360_config, model_config)

    print(f"\nAggregating metrics for method: {cfg.method_name}")
    aggregate_metrics(metrics_dir / cfg.method_name)
