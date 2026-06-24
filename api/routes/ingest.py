# api/routes/ingest.py
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.schemas import IngestResponse
from ingestion.grid_generator import generate_india_grid
from ingestion.pipeline import run_spatial_grid_pipeline
from proxy_model.predict import run_proxy_inference
from database.db_client import write_spatial_grid_batch, compute_and_store_aqi
from logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

async def _process_grid_batch_async(batch_df, lookback_days):
    """ Runs pipeline operations safely inside concurrent thread pools """
    loop = asyncio.get_running_loop()
    tasks = []
    
    for _, row in batch_df.iterrows():
        task = loop.run_in_executor(None, run_spatial_grid_pipeline, row, lookback_days)
        tasks.append(task)
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_dfs = []
    
    for df_result in results:
        if isinstance(df_result, Exception) or df_result.empty:
            continue
        try:
            inferred_df = run_proxy_inference(df_result)
            valid_dfs.append(inferred_df)
        except Exception as e:
            logger.error(f"Inference error on spatial node matrix: {e}")

    if valid_dfs:
        combined = __import__('pandas').concat(valid_dfs)
        write_spatial_grid_batch(combined)
        
        for _, row in batch_df.iterrows():
            try:
                compute_and_store_aqi(row['loc_hash'])
            except Exception:
                pass

async def _run_pan_india_grid_task(resolution: float, lookback_days: int):
    logger.info(f"Starting scheduled task loop for complete Pan India Grid infrastructure...")
    try:
        grid_df = generate_india_grid(resolution)
    except Exception as err:
        logger.error(f"Grid assembly failed entirely: {err}")
        return

    batch_size = 40  
    for i in range(0, len(grid_df), batch_size):
        chunk = grid_df.iloc[i : i + batch_size]
        await _process_grid_batch_async(chunk, lookback_days)
        await asyncio.sleep(1.0)
    logger.info("Pan-India grid orchestration loop fully completed.")

@router.post("/pan-india", status_code=202)
async def trigger_pan_india_ingest(
    background_tasks: BackgroundTasks,
    resolution: float = 0.5,
    lookback_days: int = 3
):
    if resolution < 0.25:
        raise HTTPException(status_code=400, detail="Resolution option is limited to a density threshold of 0.25.")
    
    background_tasks.add_task(_run_pan_india_grid_task, resolution, lookback_days)
    return {
        "status": "accepted",
        "message": f"Pan-India automation pipeline initialized successfully at {resolution}° interval grids."
    }