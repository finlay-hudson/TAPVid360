"""Resumable run state: one row per video in a parquet file, recording how far it got and why it failed."""
import asyncio
import logging
from pathlib import Path

import pandas as pd

from src.tapvid360.data_generation.download import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["id", "processed_state", "failure_reason", "clips_generated"]


async def load_or_create_state(run_state_path: Path, metadata_parquet: Path) -> pd.DataFrame | None:
    """Load an existing run state, or create one from the 360-1M metadata parquet.

    New runs keep videos with more than `config.MIN_LIKE_COUNT` likes, de-duplicated by video id.
    """
    if run_state_path.exists():
        df_run = await asyncio.to_thread(pd.read_parquet, run_state_path)
        if not all(col in df_run.columns for col in REQUIRED_COLUMNS):
            logger.error(f"{run_state_path} is missing required columns {REQUIRED_COLUMNS}.")
            return None
        logger.info(f"Resuming run with {len(df_run)} videos from {run_state_path}")
        return df_run

    df_meta = pd.read_parquet(metadata_parquet)
    df_meta = df_meta[df_meta["like_count"] > config.MIN_LIKE_COUNT]
    if df_meta.empty:
        logger.error(f"No videos with more than {config.MIN_LIKE_COUNT} likes in {metadata_parquet}.")
        return None
    df_run = df_meta.drop_duplicates(subset=["id"]).reset_index(drop=True).copy()
    df_run["processed_state"] = config.STATE_UNPROCESSED
    df_run["failure_reason"] = pd.NA
    df_run["clips_generated"] = 0
    await save_state(df_run, run_state_path)
    logger.info(f"Created run state with {len(df_run)} videos at {run_state_path}")
    return df_run


async def save_state(df_run: pd.DataFrame, run_state_path: Path, lock: asyncio.Lock | None = None):
    async def _save():
        try:
            await asyncio.to_thread(df_run.to_parquet, run_state_path)
        except Exception as e:
            logger.error(f"Failed to save run state to {run_state_path}: {e}")

    if lock is None:
        await _save()
    else:
        async with lock:
            await _save()


def get_state(df_run: pd.DataFrame, video_id: str) -> str:
    return df_run.loc[df_run["id"] == video_id, "processed_state"].iloc[0]


async def update_state(df_run: pd.DataFrame, video_id: str, new_state: str, info, lock: asyncio.Lock,
                       retry_failed_downloads: bool = False):
    """Record a video's new state. Final states are not overwritten by non-final ones.

    For STATE_SUCCESS, `info` is the number of clips kept; otherwise it is the failure reason. When retrying failed
    downloads, STATE_FAILED_DOWNLOAD is not treated as final so those videos can move on.
    """
    async with lock:
        idx = df_run.index[df_run["id"] == video_id].tolist()
        if not idx:
            logger.warning(f"Unknown video id {video_id}")
            return
        idx = idx[0]
        final_states = set(config.FINAL_STATES)
        if retry_failed_downloads:
            final_states.discard(config.STATE_FAILED_DOWNLOAD)
        if df_run.loc[idx, "processed_state"] in final_states and new_state not in final_states:
            return
        df_run.loc[idx, "processed_state"] = new_state
        if new_state == config.STATE_SUCCESS:
            df_run.loc[idx, "clips_generated"] = int(info) if info is not None else pd.NA
            df_run.loc[idx, "failure_reason"] = pd.NA
        else:
            df_run.loc[idx, "failure_reason"] = info if info is not None else pd.NA
            df_run.loc[idx, "clips_generated"] = pd.NA


def videos_to_process(df_run: pd.DataFrame, retry_failed_downloads: bool) -> list[str]:
    states = [config.STATE_UNPROCESSED, config.STATE_DOWNLOADED, config.STATE_PASSED_COARSE,
              config.STATE_CLIPS_GENERATED]
    if retry_failed_downloads:
        states.append(config.STATE_FAILED_DOWNLOAD)
    return df_run.loc[df_run["processed_state"].isin(states), "id"].tolist()
