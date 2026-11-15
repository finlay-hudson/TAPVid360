"""Async workers: downloaders feed a queue that filtering workers consume (download -> coarse -> clips -> fine)."""
import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

from src.tapvid360.data_generation.download import config
from src.tapvid360.data_generation.download.coarse import run_coarse_checks
from src.tapvid360.data_generation.download.fine import run_fine_checks, split_video_into_clips
from src.tapvid360.data_generation.download.state import get_state, save_state, update_state

logger = logging.getLogger(__name__)

YT_DLP_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")

# YouTube's DASH streams of 360° videos now use a "mesh" (equi-angular cubemap) layout, which the coarse checks
# reject; its HLS (m3u8) streams are still equirectangular. When TAPVid360-10k was built, `bestvideo` was
# equirectangular.
YT_DLP_FORMAT = "bestvideo[protocol^=m3u8]"


async def download_video(video_id: str, output_path: Path, cookies_path: Path | None = None) -> tuple[bool, str | None]:
    """Download the best-quality equirectangular video stream (no audio) with yt-dlp."""
    if output_path.exists():
        return True, None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["yt-dlp", f"https://youtube.com/watch?v={video_id}", "-f", YT_DLP_FORMAT,
               "-o", str(output_path), "--user-agent", YT_DLP_USER_AGENT]
    if cookies_path is not None:
        if cookies_path.exists():
            command += ["--cookies", str(cookies_path)]
        else:
            logger.warning(f"Cookies file {cookies_path} not found; downloading without it.")
    try:
        await asyncio.to_thread(subprocess.run, command, check=True, capture_output=True, text=True)
        return True, None
    except Exception as e:
        output_path.unlink(missing_ok=True)
        detail = e.stderr.strip().splitlines()[-1] if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
        return False, f"Download failed: {detail}"


class RunContext:
    """Paths, shared state and models used by all workers in one run."""

    def __init__(self, run_dir: Path, df_run, cookies_path: Path | None, retry_failed_downloads: bool,
                 lpips_model, watermark_detector, device):
        self.run_dir = run_dir
        self.state_path = run_dir / config.RUN_STATE_FILENAME
        self.download_dir = run_dir / config.TEMP_DOWNLOAD_DIR
        self.passed_coarse_dir = run_dir / config.PASSED_COARSE_DIR
        self.failed_coarse_dir = run_dir / config.FAILED_COARSE_DIR
        self.clips_dir = run_dir / config.CLIPS_DIR
        self.filtered_clips_dir = run_dir / config.FILTERED_CLIPS_DIR
        self.failed_fine_dir = run_dir / config.FAILED_FINE_DIR
        for d in (self.download_dir, self.passed_coarse_dir, self.failed_coarse_dir, self.clips_dir,
                  self.filtered_clips_dir, self.failed_fine_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.df_run = df_run
        self.cookies_path = cookies_path
        self.retry_failed_downloads = retry_failed_downloads
        self.lpips_model = lpips_model
        self.watermark_detector = watermark_detector
        self.device = device
        self.state_lock = asyncio.Lock()
        self.save_lock = asyncio.Lock()

    async def set_state(self, video_id, new_state, info=None):
        if info is not None and new_state != config.STATE_SUCCESS:
            info = str(info)  # Failure reasons are stored as text so the parquet file stays writable.
        await update_state(self.df_run, video_id, new_state, info, self.state_lock, self.retry_failed_downloads)

    async def get_state(self, video_id):
        async with self.state_lock:
            return get_state(self.df_run, video_id)

    async def save(self):
        await save_state(self.df_run, self.state_path, self.save_lock)


async def download_worker(name: str, ids_queue: asyncio.Queue, process_queue: asyncio.Queue, ctx: RunContext):
    """Download unprocessed videos; videos part-way through a resumed run go straight to processing."""
    while True:
        video_id = await ids_queue.get()
        try:
            state = await ctx.get_state(video_id)
            needs_download = state == config.STATE_UNPROCESSED or (
                    ctx.retry_failed_downloads and state == config.STATE_FAILED_DOWNLOAD)
            if not needs_download:
                await process_queue.put(video_id)
                continue
            ok, reason = await download_video(video_id, ctx.download_dir / f"{video_id}.mp4", ctx.cookies_path)
            if ok:
                await ctx.set_state(video_id, config.STATE_DOWNLOADED)
                await process_queue.put(video_id)
            else:
                logger.info(f"[{name}] {video_id}: {reason}")
                await ctx.set_state(video_id, config.STATE_FAILED_DOWNLOAD, reason)
            await ctx.save()
        except Exception as e:
            logger.error(f"[{name}] {video_id}: unexpected error: {e}", exc_info=True)
        finally:
            ids_queue.task_done()


async def process_worker(name: str, process_queue: asyncio.Queue, ctx: RunContext):
    """Take each video through coarse filtering, 10 s clip splitting and fine filtering, saving state after each."""
    while True:
        video_id = await process_queue.get()
        if video_id is None:
            process_queue.task_done()
            break
        try:
            if await ctx.get_state(video_id) == config.STATE_DOWNLOADED:
                video_path = ctx.download_dir / f"{video_id}.mp4"
                try:
                    passed, reason = await asyncio.to_thread(run_coarse_checks, video_path, ctx.lpips_model, ctx.device)
                except Exception as e:
                    passed, reason = False, f"Unexpected coarse processing error: {e}"
                if not passed:
                    logger.info(f"[{name}] {video_id} failed coarse filtering: {reason}")
                    await ctx.set_state(video_id, config.STATE_FAILED_COARSE, reason)
                    if config.STORE_FAILED:
                        await asyncio.to_thread(shutil.move, str(video_path), str(ctx.failed_coarse_dir / video_path.name))
                    else:
                        video_path.unlink(missing_ok=True)
                    await ctx.save()
                    continue
                await asyncio.to_thread(shutil.move, str(video_path), str(ctx.passed_coarse_dir / video_path.name))
                await ctx.set_state(video_id, config.STATE_PASSED_COARSE)
                await ctx.save()

            if await ctx.get_state(video_id) == config.STATE_PASSED_COARSE:
                video_path = ctx.passed_coarse_dir / f"{video_id}.mp4"
                await asyncio.to_thread(split_video_into_clips, video_path, ctx.clips_dir / video_id)
                await ctx.set_state(video_id, config.STATE_CLIPS_GENERATED)
                await ctx.save()
                video_path.unlink()

            if await ctx.get_state(video_id) == config.STATE_CLIPS_GENERATED:
                try:
                    passed, info = await asyncio.to_thread(run_fine_checks, video_id, ctx.clips_dir,
                                                           ctx.filtered_clips_dir, ctx.failed_fine_dir,
                                                           ctx.watermark_detector)
                except Exception as e:
                    passed, info = False, f"Unexpected fine processing error: {e}"
                await ctx.set_state(video_id, config.STATE_SUCCESS if passed else config.STATE_FAILED_FINE, info)
                await ctx.save()
                logger.info(f"[{name}] {video_id}: " + (f"kept {info} clips" if passed else f"failed fine filtering: {info}"))
        except Exception as e:
            logger.error(f"[{name}] {video_id}: unexpected error: {e}", exc_info=True)
        finally:
            process_queue.task_done()
