from src.tapvid360.evaluation.eval_configs import TAPVid360EvalConfig
from src.tapvid360.evaluation.metrics.metrics_utils import aggregate_metrics

if __name__ == "__main__":
    cfg = TAPVid360EvalConfig()
    metrics_dir = cfg.metrics_out_root / cfg.data_dir.stem
    method = "cotracker360"

    print(f"\nAggregating metrics for method: {method}")
    aggregate_metrics(metrics_dir / method)
