"""Fine-tune CoTracker3 into CoTracker360 on generated TAPVid360 training data.

CoTracker3 (offline) weights initialise everything except the final flow head, which is replaced to predict rotations.
Directions are supervised with a Huber loss; CoTracker3's visibility and confidence outputs are not supervised.
The model is validated on TAPVid360-10k (mean angular error) every `--val_every` epochs.

Example:
    PYTHONPATH=. python src/tapvid360/training/train.py \
        --train_dir outputs/TAPVid360-training --val_dir data/TAPVid360-10k \
        --cotracker3_checkpoint checkpoints/scaled_offline.pth --out_dir outputs/training/cotracker360

Checkpoints (`checkpoint_last.pth`, `checkpoint_best.pth`, `epoch_<n>/checkpoint.pth`) load directly with
`CoTracker360.load_weights` (and so with the evaluation code).
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tapvid360.common.model.cotracker360 import CoTracker360
from src.tapvid360.common.model.utils import CheckpointSaver
from src.tapvid360.evaluation.data.data_utils import collate_fn, get_transform
from src.tapvid360.evaluation.data.eval_dataloader import TAPVid360Dataset
from src.tapvid360.evaluation.metrics.tapvid360_metrics import calc_angular_dist
from src.tapvid360.evaluation.model_runner import _get_tapvid360_mapper, _make_query_points_for_cotracker

# CoTracker360 architecture settings (as in evaluation/eval_configs.py).
MODEL_RESOLUTION = (240, 320)
STRIDE = 4
CORR_RADIUS = 3
WINDOW_LEN = 60
# Replaced so the decoder predicts a rotation per point and frame.
REINITIALISED_LAYERS = ["updateformer.flow_head.weight", "updateformer.flow_head.bias"]


def forward(model, batch, device, num_iters):
    """Run CoTracker360 on a batch. Returns (predicted unit vectors (B, T, N, 3), Huber loss)."""
    batch.to_device(device)
    mappers = _get_tapvid360_mapper(batch.equirectangular_height, batch.video.shape[-1], batch.video.shape[-2],
                                    batch.fov_x)
    query_points = torch.stack([mapper.point.vc.to_ij(batch.trajectory[b]) for b, mapper in enumerate(mappers)])
    query_points = _make_query_points_for_cotracker(batch.trajectory, query_points)
    coord_preds, _vis, _conf, _train_data = model(batch.video, all_queries=query_points, mapper=mappers,
                                                  iters=num_iters, is_train=True)
    loss = torch.nn.functional.huber_loss(coord_preds, batch.trajectory, reduction="none").sum(dim=-1).mean()
    return coord_preds, loss


def train_one_epoch(model, loader, optimizer, device, num_iters, epoch):
    model.train()
    losses = []
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for batch in progress:
        _, loss = forward(model, batch, device, num_iters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return losses


@torch.no_grad()
def validate(model, loader, device, num_iters, epoch):
    """Angular error (degrees) of every point and frame of the validation set."""
    model.eval()
    dists = []
    progress = tqdm(loader, desc=f"Epoch {epoch} [val]")
    for batch in progress:
        coord_preds, _ = forward(model, batch, device, num_iters)
        dists.append(calc_angular_dist(coord_preds, batch.trajectory).cpu())
        progress.set_postfix(mean_dist=f"{torch.cat(dists).mean():.4f}")
    return torch.cat(dists)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = CoTracker360(model_resolution=MODEL_RESOLUTION, window_len=WINDOW_LEN, corr_radius=CORR_RADIUS,
                         stride=STRIDE)
    model.load_weights(args.cotracker3_checkpoint, ignored_layers=REINITIALISED_LAYERS)
    model = model.to(device)

    transform = get_transform(model.model_resolution)
    train_dataset = TAPVid360Dataset(args.train_dir, transform=transform, num_frames=args.num_frames,
                                     num_queries=args.num_queries, debug_max_items=args.max_train_items)
    val_dataset = TAPVid360Dataset(args.val_dir, transform=transform, num_frames=args.num_frames,
                                   num_queries=args.num_val_queries, debug_max_items=args.max_val_items)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn,
                         pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"{len(train_dataset)} training and {len(val_dataset)} validation samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type == "cuda")
    checkpointer = CheckpointSaver(args.out_dir)
    losses = {"train": {}, "val": {}, "test": {}}
    dists = {"train": {}, "val": {}, "test": {}}
    start_epoch = 0
    if args.resume:
        checkpoint = checkpointer.load("checkpoint_last.pth")
        model.load_weights(args.out_dir / "checkpoint_last.pth", strict=True)
        losses, dists = checkpoint["extras"]["losses"], checkpoint["extras"]["dists"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    with open(args.out_dir / "args.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        losses["train"][epoch] = [{"flow": loss} for loss in train_one_epoch(model, train_loader, optimizer, device,
                                                                            args.num_iters, epoch)]
        mean_loss = torch.stack([loss["flow"] for loss in losses["train"][epoch]]).mean().item()
        print(f"Epoch {epoch}: mean train loss {mean_loss:.4f}")
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            dists["val"][epoch] = validate(model, val_loader, device, args.num_iters, epoch)
            print(f"Epoch {epoch}: mean val angular error {dists['val'][epoch].mean():.4f} deg")
            model.save_checkpoint(checkpointer, epoch, dists, losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_dir", type=Path, required=True,
                        help="Directory containing dataset/ of training samples in TAPVid360 format "
                             "(data_generation/sequences/main.py --mode training --out_dir <train_dir>/dataset).")
    parser.add_argument("--val_dir", type=Path, required=True, help="TAPVid360-10k root (containing dataset/).")
    parser.add_argument("--cotracker3_checkpoint", type=Path, required=True,
                        help="CoTracker3 offline weights (scaled_offline.pth).")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=1, help="CoTracker refinement iterations.")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_queries", type=int, default=32, help="Query points per training sample.")
    parser.add_argument("--num_val_queries", type=int, default=256)
    parser.add_argument("--val_every", type=int, default=5, help="Validate and checkpoint every N epochs.")
    parser.add_argument("--max_train_items", type=int, default=-1, help="Limit training samples (-1: all).")
    parser.add_argument("--max_val_items", type=int, default=-1, help="Limit validation samples (-1: all).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Continue from <out_dir>/checkpoint_last.pth.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
