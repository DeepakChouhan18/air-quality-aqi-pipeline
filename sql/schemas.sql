-- ==========================================================
-- SCHEMA LAYOUT
-- raw       -> exact copies of ingested records (audit trail)
-- staging   -> parsed + validated records, not yet aggregated
-- core      -> clean, deduplicated, business-ready tables
-- analytics -> aggregated gold-layer tables for the dashboard
-- ==========================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Pipeline lineage / audit table (lives in staging schema)
-- Every ingestion + transform run writes one row here.
CREATE TABLE IF NOT EXISTS staging.pipeline_runs (
    run_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    stage TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,          -- RUNNING / SUCCESS / FAILED
    rows_extracted INTEGER DEFAULT 0,
    rows_valid INTEGER DEFAULT 0,
    rows_rejected INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS staging.rejected_records (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES staging.pipeline_runs(run_id),
    source TEXT,
    original_record JSONB,
    rejection_reason TEXT,
    rejected_at TIMESTAMPTZ DEFAULT now()
);
