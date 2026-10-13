# scheduler.py
"""
Local stand-in for the GitHub Actions cron: asks a running API to run the
pan-India pipeline every 3 hours (the cadence the free Open-Meteo quota allows).
"""

import time

import requests

API_URL = "http://localhost:8000/ingest/pan-india?resolution=1.0"
INTERVAL_S = 3 * 3600

print("⏰ Pan-India AQI automation daemon started — triggering the pipeline every 3 hours.")

while True:
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Triggering pan-India pipeline...")
        response = requests.post(API_URL, timeout=30)
        if response.status_code == 202:
            print("✅ Pipeline job accepted by the API.")
        else:
            print(f"⚠️ Server responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed (is the API running?): {e}")

    time.sleep(INTERVAL_S)
