"""
Extracts air-pollution data from the OpenAQ v3 API and saves it,
untouched, into the raw landing zone.

WHY WE SAVE RAW DATA BEFORE TRANSFORMING:
If a transformation bug is discovered next week, you can re-run it
against the original API response instead of re-calling the API (which
may have changed or rate-limited you). Raw data is your source of
truth and audit trail.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import OPENAQ_API_KEY, OPENAQ_BASE_URL, TARGET_CITY, TARGET_LAT, TARGET_LON, RAW_DIR
from src.ingestion.ingestion_utils import fetch_with_retry, PipelineRun


def get_locations(limit: int = 100) -> dict:
    """
    Fetch monitoring station/location metadata near the target city.

    WHY COORDINATES INSTEAD OF city= PARAM:
    OpenAQ v3 removed the `city` string filter from /v3/locations because
    city names were inconsistently labeled across data providers. The v3
    replacement is coordinate-based (lat/lon + radius in metres). We search
    within 25 km of the configured city centre, which reliably captures all
    urban monitoring stations.
    """
    headers = {"X-API-Key": OPENAQ_API_KEY}
    params = {
        "coordinates": f"{TARGET_LAT},{TARGET_LON}",
        "radius": 25000,   # 25 km radius covers the full urban area
        "limit": limit,
    }
    response = fetch_with_retry(
        f"{OPENAQ_BASE_URL}/locations", params=params, headers=headers
    )
    return response.json()


def get_latest_measurements(location_id: int) -> dict:
    """Fetch the latest pollutant measurements for one station."""
    headers = {"X-API-Key": OPENAQ_API_KEY}
    response = fetch_with_retry(
        f"{OPENAQ_BASE_URL}/locations/{location_id}/latest", headers=headers
    )
    return response.json()


def run_openaq_ingestion() -> Path:
    """
    Full ingestion run: fetch stations + latest measurements for the
    target city, save raw JSON, and log lineage.

    WHY WE ALSO SAVE A SENSOR MAP:
    The OpenAQ v3 /locations/{id}/latest endpoint returns measurement rows
    keyed only by sensorsId (an integer). It does NOT include the pollutant
    name (pm25, no2 etc.) or units in each measurement row — those live in
    the /locations response under locations[].sensors[].parameter.
    We extract a sensorsId -> {parameter, unit} lookup here and save it as
    sensors_map.json so the transformation step can join it without re-calling
    the API. This is the standard "enrich-at-ingest" pattern.

    Returns the path to the saved measurements file.
    """
    run = PipelineRun(source="openaq", stage="extract")
    today = datetime.now(timezone.utc).date().isoformat()
    out_dir = RAW_DIR / "openaq" / today
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        locations = get_locations()
        with open(out_dir / "locations.json", "w") as f:
            json.dump(locations, f, indent=2)

        # Build sensor_id -> {parameter_name, unit} lookup from the locations response.
        # The /latest endpoint only returns sensorsId; without this map we cannot
        # know which pollutant each value belongs to.
        sensors_map = {}
        for loc in locations.get("results", []):
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                sensors_map[sensor["id"]] = {
                    "parameter": param.get("name"),
                    "unit": param.get("units"),
                }

        with open(out_dir / "sensors_map.json", "w") as f:
            json.dump(sensors_map, f, indent=2)

        all_measurements = []
        for loc in locations.get("results", []):
            measurements = get_latest_measurements(loc["id"])
            all_measurements.append(measurements)

        measurements_path = out_dir / "measurements.json"
        with open(measurements_path, "w") as f:
            json.dump(all_measurements, f, indent=2)

        total_rows = sum(len(m.get("results", [])) for m in all_measurements)
        run.mark_success(rows_extracted=total_rows)
        print(f"[openaq] Saved {total_rows} measurements from {len(all_measurements)} stations -> {measurements_path}")
        print(f"[openaq] Sensor map: {len(sensors_map)} sensors -> {out_dir / 'sensors_map.json'}")
        return measurements_path

    except Exception as e:
        run.mark_failure(e)
        print(f"[openaq] Ingestion FAILED: {e}")
        raise


if __name__ == "__main__":
    run_openaq_ingestion()
