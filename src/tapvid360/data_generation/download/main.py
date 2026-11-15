"""Stage 1: download 360-1M videos and filter them into 10-second 360° clips.

Example:
    PYTHONPATH=. python src/tapvid360/data_generation/download/main.py \
        --metadata path/to/360-1M.parquet --out_dir outputs/download

Outputs go to `<out_dir>/<run name>/filtered_clips/<video_id>/<video_id>_clip_NNNN.mp4`. Runs are resumable: pass
`--resume_dir <out_dir>/<run name>` to continue an interrupted run.
"""
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import lpips
import pandas as pd

from src.tapvid360.data_generation.download import config
from src.tapvid360.data_generation.download.fine import WatermarkDetector
from src.tapvid360.data_generation.download.state import load_or_create_state, save_state, videos_to_process
from src.tapvid360.data_generation.download.workers import RunContext, download_worker, process_worker

logger = logging.getLogger(__name__)


async def log_progress(ctx: RunContext, wandb_run, interval_s: int = 60):
    """Periodically log how many videos are in each state."""
    while True:
        await asyncio.sleep(interval_s)
        async with ctx.state_lock:
            counts = ctx.df_run["processed_state"].value_counts().to_dict()
            clips = int(pd.to_numeric(ctx.df_run["clips_generated"], errors="coerce").fillna(0).sum())
        log = {"total_processed": sum(n for s, n in counts.items() if s in config.FINAL_STATES),
               "successful": counts.get(config.STATE_SUCCESS, 0), "filtered_clips_total": clips,
               **{f"state_{s}": counts.get(s, 0) for s in config.TRACKED_STATES}}
        logger.info(f"Progress: {log}")
        if wandb_run is not None:
            wandb_run.log(log)


async def main(args):
    if args.resume_dir is not None:
        run_dir = args.resume_dir
    else:
        run_dir = args.out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    df_run = await load_or_create_state(run_dir / config.RUN_STATE_FILENAME, args.metadata)
    if df_run is None:
        return
    video_ids = videos_to_process(df_run, args.retry_failed_downloads)
    if not video_ids:
        logger.info("Nothing left to process.")
        return
    logger.info(f"{len(video_ids)} videos to process.")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="tapvid360-download", id=run_dir.name, resume="allow")

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()
    watermark_detector = WatermarkDetector(device=args.device)
    ctx = RunContext(run_dir, df_run, args.cookies, args.retry_failed_downloads, lpips_model, watermark_detector,
                     args.device)

    ids_queue, process_queue = asyncio.Queue(), asyncio.Queue(maxsize=config.VIDEO_QUEUE_SIZE)
    for video_id in video_ids:
        ids_queue.put_nowait(video_id)

    downloaders = [asyncio.create_task(download_worker(f"Downloader-{i + 1}", ids_queue, process_queue, ctx))
                   for i in range(config.MAX_DOWNLOAD_WORKERS)]
    processors = [asyncio.create_task(process_worker(f"Processor-{i + 1}", process_queue, ctx))
                  for i in range(config.MAX_PROCESS_WORKERS)]
    progress = asyncio.create_task(log_progress(ctx, wandb_run))

    await ids_queue.join()
    for _ in processors:
        await process_queue.put(None)
    await asyncio.gather(*processors)
    for task in downloaders + [progress]:
        task.cancel()
    await asyncio.gather(*downloaders, progress, return_exceptions=True)

    await save_state(ctx.df_run, ctx.state_path)
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, required=True,
                        help="360-1M metadata parquet (needs at least 'id' and 'like_count' columns).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Base directory; each run gets a sub-directory.")
    parser.add_argument("--resume_dir", type=Path, default=None, help="Existing run directory to resume.")
    parser.add_argument("--cookies", type=Path, default=None, help="Optional cookies file for yt-dlp.")
    parser.add_argument("--retry_failed_downloads", action="store_true", help="Retry videos that failed to download.")
    parser.add_argument("--device", default="cpu", help="Device for LPIPS and the watermark model (default: cpu).")
    parser.add_argument("--wandb", action="store_true", help="Log progress to Weights & Biases.")
    asyncio.run(main(parser.parse_args()))
