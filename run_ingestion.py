"""
run_ingestion.py
----------------
Backwards-compatible entry point (used by auto_ingest.bat / Windows Task
Scheduler). Runs one full pan-India pass: ingest → forecast → skill → prune.
Equivalent to: python run_pipeline.py all
"""

import run_pipeline

if __name__ == "__main__":
    run_pipeline.main(["all"])
