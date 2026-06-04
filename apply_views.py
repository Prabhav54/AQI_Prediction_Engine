import os
from database.db_client import get_sync_engine
from sqlalchemy import text

def inject_sql_views():
    sql_file_path = os.path.join("database", "aqi_sql.sql")
    
    if not os.path.exists(sql_file_path):
        print(f"❌ Could not find SQL file at: {sql_file_path}")
        return

    print(f"📖 Reading {sql_file_path}...")
    with open(sql_file_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    print("🔌 Connecting to TimescaleDB via project engine...")
    engine = get_sync_engine()
    
    try:
        with engine.connect() as conn:
            # Wrap the entire script in a text block and execute
            conn.execute(text(sql_script))
            conn.commit()
        print("✅ Success! All database views and calculations applied perfectly.")
    except Exception as e:
        print(f"❌ Database execution failed: {e}")

if __name__ == "__main__":
    inject_sql_views()