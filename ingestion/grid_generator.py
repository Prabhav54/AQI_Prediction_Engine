# ingestion/grid_generator.py
import geopandas as gpd
import numpy as np
import pandas as pd
import hashlib

def generate_india_grid(resolution_deg: float = 0.5) -> pd.DataFrame:
    """
    Generate a grid of points covering India's landmass using dynamic geometry boundaries.
    0.5° ≈ 55km resolution → ~1,400 points.
    0.25° ≈ 27km resolution → ~5,500 points.
    """
    try:
        india = gpd.read_file(
            "https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson"
        ).to_crs("EPSG:4326")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch India shapefile map from source: {e}")

    india_union = india.geometry.union_all()

    lats = np.arange(6.5, 37.5, resolution_deg)
    lons = np.arange(68.5, 97.5, resolution_deg)

    points = []
    for lat in lats:
        for lon in lons:
            pt = __import__('shapely.geometry').geometry.Point(lon, lat)
            if india_union.contains(pt):
                lat_r = round(lat, 4)
                lon_r = round(lon, 4)
                loc_hash = hashlib.md5(f"{round(lat_r,3)},{round(lon_r,3)}".encode()).hexdigest()[:8]
                points.append({
                    "lat": lat_r,
                    "lon": lon_r,
                    "loc_hash": loc_hash,
                    "location_name": f"Grid Node ({lat_r}N, {lon_r}E)"
                })

    df = pd.DataFrame(points)
    print(f"Grid complete: Generated {len(df)} points inside India at {resolution_deg}° resolution.")
    return df