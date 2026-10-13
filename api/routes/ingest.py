# api/routes/ingest.py
"""
Manual trigger for a full pipeline pass (ingest → forecast → skill → prune).
In production the same job runs on a GitHub Actions cron; this endpoint is
for local runs and the Streamlit "Sync" button.
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import IngestResponse
from logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_lock = asyncio.Lock()


async def _run_pipeline(resolution: float) -> None:
    import run_pipeline

    if _lock.locked():
        logger.warning("Pipeline already running — skipping trigger.")
        return
    async with _lock:
        try:
            await asyncio.to_thread(run_pipeline.main, ["all", "--resolution", str(resolution)])
        except Exception as exc:
            logger.error("Pipeline run failed: {}", exc)


@router.post("/pan-india", status_code=202, response_model=IngestResponse)
async def trigger_pan_india_ingest(background_tasks: BackgroundTasks, resolution: float = 1.0):
    if resolution not in (0.5, 1.0):
        raise HTTPException(status_code=400, detail="Committed grids exist for 0.5° and 1.0° only.")
    background_tasks.add_task(_run_pipeline, resolution)
    return IngestResponse(status="accepted",
                          message=f"Pan-India pipeline started on the {resolution}° grid + city nodes.")
