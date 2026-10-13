"""
ingestion/grid_generator.py
---------------------------
Builds the pan-India node table the whole system runs on.

Two node kinds:
  * grid — a uniform lat/lon lattice clipped to India's landmass
  * city — every entry in config/cities.py, so searched-for places get a
           forecast centred on them

Clipping needs geopandas + the India boundary polygon, which is slow and heavy.
So the clipped lattice is generated once and committed as a CSV under
ingestion/nodes/. The hourly job only reads that CSV — no geopandas, no
network call. Regenerate with:

    python -m ingestion.grid_generator --resolution 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.cities import MAJOR_CITIES
from config.settings import GRID_RESOLUTION_DEG, INDIA_BOUNDARY_URL
from logger import get_logger
from utils import location_hash

logger = get_logger(__name__)

NODES_DIR = Path(__file__).parent / "nodes"

# India bounding box (incl. islands)
LAT_MIN, LAT_MAX = 6.5, 37.5
LON_MIN, LON_MAX = 68.0, 97.5

# Grid nodes closer than this to a city node are dropped (city node wins)
CITY_DEDUP_KM = 25.0


def _grid_csv_path(resolution_deg: float) -> Path:
    return NODES_DIR / f"india_grid_{resolution_deg:.2f}.csv"


def generate_india_grid(resolution_deg: float = 0.5, boundary: str = INDIA_BOUNDARY_URL) -> pd.DataFrame:
    """
    Generate a lattice of points inside India's boundary polygon.
    0.5° ≈ 55 km → ~1,150 points.  1.0° ≈ 110 km → ~290 points.

    `boundary` can be a URL or a local GeoJSON path.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    try:
        india = gpd.read_file(boundary).to_crs("EPSG:4326")
    except Exception as e:
        raise RuntimeError(f"Failed to read India boundary from {boundary}: {e}")

    india_union = india.geometry.union_all()

    # Offset the lattice by half a cell so points sit at cell centres
    lats = np.arange(LAT_MIN + resolution_deg / 2, LAT_MAX, resolution_deg)
    lons = np.arange(LON_MIN + resolution_deg / 2, LON_MAX, resolution_deg)

    points = []
    for lat in lats:
        for lon in lons:
            if india_union.contains(Point(lon, lat)):
                points.append({"lat": round(float(lat), 4), "lon": round(float(lon), 4)})

    df = pd.DataFrame(points)
    logger.info("Grid complete: {} points inside India at {}°", len(df), resolution_deg)
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_node_table(resolution_deg: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """
    Return every node the pipeline tracks:
        location_hash, location_name, state, lat, lon, kind ('city' | 'grid')

    Grid nodes are labelled with their nearest city, e.g. "Grid 23.5N 80.5E (near Jabalpur)".
    """
    csv_path = _grid_csv_path(resolution_deg)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No grid file for {resolution_deg}° at {csv_path}. "
            f"Run: python -m ingestion.grid_generator --resolution {resolution_deg}"
        )
    grid = pd.read_csv(csv_path)

    cities = pd.DataFrame(MAJOR_CITIES, columns=["location_name", "state", "lat", "lon"])
    cities["kind"] = "city"

    # Distance from every grid node to every city: (n_grid, n_city)
    d = _haversine_km(
        grid["lat"].values[:, None], grid["lon"].values[:, None],
        cities["lat"].values[None, :], cities["lon"].values[None, :],
    )
    nearest = d.argmin(axis=1)
    grid = grid.loc[d.min(axis=1) > CITY_DEDUP_KM].copy()
    nearest = nearest[d.min(axis=1) > CITY_DEDUP_KM]

    grid["state"] = cities["state"].values[nearest]
    grid["location_name"] = [
        f"Grid {lat:.2f}N {lon:.2f}E (near {cities['location_name'].iat[i]})"
        for lat, lon, i in zip(grid["lat"], grid["lon"], nearest)
    ]
    grid["kind"] = "grid"

    nodes = pd.concat([cities, grid], ignore_index=True)
    nodes["location_hash"] = [location_hash(a, b) for a, b in zip(nodes["lat"], nodes["lon"])]
    nodes = nodes.drop_duplicates("location_hash").reset_index(drop=True)
    return nodes[["location_hash", "location_name", "state", "lat", "lon", "kind"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the clipped India grid CSV.")
    parser.add_argument("--resolution", type=float, default=GRID_RESOLUTION_DEG)
    parser.add_argument("--boundary", default=INDIA_BOUNDARY_URL,
                        help="URL or local path of the India boundary GeoJSON")
    args = parser.parse_args()

    NODES_DIR.mkdir(exist_ok=True)
    out = _grid_csv_path(args.resolution)
    generate_india_grid(args.resolution, args.boundary).to_csv(out, index=False)
    nodes = build_node_table(args.resolution)
    print(f"Wrote {out} | total nodes incl. cities: {len(nodes)} "
          f"({(nodes.kind == 'city').sum()} cities, {(nodes.kind == 'grid').sum()} grid)")
