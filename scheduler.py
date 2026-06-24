# scheduler.py
import time
import requests

API_URL = "http://localhost:8000/ingest/pan-india?resolution=0.5&lookback_days=1"

print("⏰ Pan-India AQI Automation Daemon Started.")
print("The pipeline will trigger automatically at the top of every hour.")

while True:
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Triggering grid ingestion pipeline...")
        response = requests.post(API_URL, timeout=30)
        if response.status_code == 202:
            print("✅ Pipeline job successfully accepted by backend workers.")
        else:
            print(f"⚠️ Server responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed (Is your Uvicorn server running?): {e}")
    
    # Sleep exactly 1 hour (3600 seconds)
    time.sleep(3600)