"""
Shared ingestion helpers: retry logic and pipeline lineage/audit logging.

WHY LINEAGE MATTERS (viva-ready answer):
Every run of the pipeline should leave a trail answering: which source
ran, when, did it succeed, how many rows came in, how many passed
validation, how many were rejected, and what error (if any) occurred.
Without this, debugging a bad day of data means guessing. With it,
you query one table and know exactly what happened.

We log to BOTH:
  1. A local CSV (data/logs/pipeline_runs.csv) so it works even before
     Postgres is set up.
  2. Postgres `staging.pipeline_runs` once the database exists (used by
     the Airflow DAG stage, added in load_postgres.py).
"""

import csv
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from src.config import LOGS_DIR

PIPELINE_LOG_CSV = LOGS_DIR / "pipeline_runs.csv"

CSV_FIELDS = [
    "run_id",
    "source",
    "stage",
    "started_at",
    "finished_at",
    "status",
    "rows_extracted",
    "rows_valid",
    "rows_rejected",
    "error_message",
]


@dataclass
class PipelineRun:
    """Represents one lineage record for one stage of one source."""

    source: str
    stage: str  # e.g. "extract", "validate", "transform", "load"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    status: str = "RUNNING"
    rows_extracted: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    error_message: str = ""

    def mark_success(self, rows_extracted=0, rows_valid=0, rows_rejected=0):
        self.status = "SUCCESS"
        self.rows_extracted = rows_extracted
        self.rows_valid = rows_valid
        self.rows_rejected = rows_rejected
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self._write()

    def mark_failure(self, error: Exception):
        self.status = "FAILED"
        self.error_message = str(error)
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self._write()

    def _write(self):
        """Append this run record to the CSV lineage log."""
        file_exists = PIPELINE_LOG_CSV.exists()
        with open(PIPELINE_LOG_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(self))


def fetch_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: int = 30,
) -> requests.Response:
    """
    Call an API with automatic retries and exponential backoff.

    WHY: Free public APIs (OpenAQ, Open-Meteo) occasionally time out or
    rate-limit. Failing immediately on the first hiccup wastes a whole
    scheduled run. Retrying with increasing delay handles transient
    issues without hammering the API.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                wait = backoff_seconds * attempt
                print(f"[retry {attempt}/{max_retries}] {url} failed: {e}. "
                      f"Retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise last_exception
