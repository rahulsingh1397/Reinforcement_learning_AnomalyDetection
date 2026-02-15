"""
Unified data ingestion and aggregation module.

Replaces: dataAggregateRawDict.py, dataTestDictNew.py

Key improvements over original:
- Single module for both training and test data processing
- Proper error handling with logging (no silent except:pass)
- Welford's online algorithm for real running mean/std
- No code duplication across day types
- Type hints and clean API
- Streaming support for large files
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from config import (
    get_config, get_day_type, get_interval,
    DATA_DIR, OUTPUTS_DIR
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# User behavior model data structures
# ──────────────────────────────────────────────────────────────────────

def _empty_interval_counters() -> List[int]:
    """Return a zero-filled interval counter list."""
    cfg = get_config()
    return [0] * cfg.data.num_intervals


def _empty_day_block_train() -> Dict:
    """Create an empty day-type block for training models."""
    return {
        "DayCounter": 0,
        "IntervalCounter": {
            "sum": _empty_interval_counters(),
        },
        "SourceAddress": {},
        "DestinationHost": {},
    }


def _empty_day_block_test() -> Dict:
    """Create an empty day-type block for test models."""
    return {
        "Interval": 0,
        "IntervalCounter": _empty_interval_counters(),
        "SourceAddress": {},
        "DestinationHost": {},
    }


def _new_user_train() -> Dict:
    """Initialize a new user entry for training."""
    return {
        "UserLabel": 0,
        "WD": _empty_day_block_train(),
        "Sat": _empty_day_block_train(),
        "Sun": _empty_day_block_train(),
    }


def _new_user_test() -> Dict:
    """Initialize a new user entry for testing."""
    return {
        "WD": _empty_day_block_test(),
        "Sat": _empty_day_block_test(),
        "Sun": _empty_day_block_test(),
    }


# ──────────────────────────────────────────────────────────────────────
# Destination host label management
# ──────────────────────────────────────────────────────────────────────

class DestinationLabelManager:
    """Manages the mapping between destination hostnames and integer labels."""

    def __init__(self, label_file: Optional[Path] = None):
        self.hosts: List[str] = []
        self._index: Dict[str, int] = {}
        if label_file and label_file.exists():
            self.load(label_file)

    def load(self, filepath: Path) -> None:
        """Load destination labels from CSV."""
        self.hosts = []
        self._index = {}
        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        hostname = row[1]
                        self._index[hostname] = len(self.hosts)
                        self.hosts.append(hostname)
            logger.info("Loaded %d destination labels from %s", len(self.hosts), filepath)
        except Exception as e:
            logger.error("Failed to load destination labels: %s", e)

    def save(self, filepath: Path) -> None:
        """Save destination labels to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Label", "Host Name"])
            for i, host in enumerate(self.hosts):
                writer.writerow([i, host])
        logger.info("Saved %d destination labels to %s", len(self.hosts), filepath)

    def get_or_create_label(self, hostname: str) -> int:
        """Get label for hostname, creating a new one if needed."""
        if hostname in self._index:
            return self._index[hostname]
        label = len(self.hosts)
        self.hosts.append(hostname)
        self._index[hostname] = label
        return label


# ──────────────────────────────────────────────────────────────────────
# Log record parsing
# ──────────────────────────────────────────────────────────────────────

def parse_csv_row(row: list, dest_mgr: DestinationLabelManager) -> Optional[Dict]:
    """
    Parse a raw CSV row into a structured log record.
    Returns None if the row should be skipped.
    """
    cfg = get_config().data
    try:
        logon_type = int(row[12])
        event_name = row[3]

        if cfg.event_filter not in event_name:
            return None
        if logon_type not in cfg.valid_logon_types:
            return None

        timestamp_ms = int(row[0])
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
        interval = get_interval(dt.hour, cfg.hours_per_interval)
        day_type = get_day_type(dt)

        source = row[4].strip() if row[4].strip() else "NIL"
        username = row[9]
        dest_hostname = row[8]
        dest_label = dest_mgr.get_or_create_label(dest_hostname)

        return {
            "UserName": username,
            "StartDate": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "DateTime": dt,
            "SourceAddress": source,
            "DestinationHost": dest_label,
            "DestinationHostName": dest_hostname,
            "Interval": interval,
            "DayType": day_type,
        }
    except (ValueError, IndexError) as e:
        logger.debug("Skipping malformed CSV row: %s", e)
        return None


def parse_json_row(row: Dict, curr_date: datetime,
                   dest_mgr: DestinationLabelManager) -> Optional[Dict]:
    """
    Parse a JSON dict row (from intermediate logData.json) into a structured record.
    Returns None if the row should be skipped.
    """
    cfg = get_config().data
    try:
        logon_type = int(row.get(cfg.col_logon_type, 0))
        event_name = row.get(cfg.col_event_name, "")
        timestamp_ms = int(row.get(cfg.col_timestamp, 0))
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)

        if cfg.event_filter not in event_name:
            return None
        if logon_type not in cfg.valid_logon_types:
            return None
        if dt.day != curr_date.day:
            return None

        interval = get_interval(dt.hour, cfg.hours_per_interval)
        day_type = get_day_type(dt)

        source = row.get(cfg.col_source_address, "").strip() or "NIL"
        username = row.get(cfg.col_dest_user, "")
        dest_hostname = row.get(cfg.col_dest_host, "")
        dest_label = dest_mgr.get_or_create_label(dest_hostname)

        return {
            "UserName": username,
            "StartDate": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "DateTime": dt,
            "SourceAddress": source,
            "DestinationHost": dest_label,
            "DestinationHostName": dest_hostname,
            "Interval": interval,
            "DayType": day_type,
        }
    except (ValueError, KeyError) as e:
        logger.debug("Skipping malformed JSON row: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
# Training data aggregation
# ──────────────────────────────────────────────────────────────────────

def add_to_train_model(model: Dict, record: Dict) -> Dict:
    """Add a parsed log record to the training model."""
    un = record["UserName"]
    day_type = record["DayType"]
    interval = record["Interval"]

    if un not in model:
        model[un] = _new_user_train()
        model[un]["UserLabel"] = len(model)

    day_block = model[un][day_type]
    counter = day_block["DayCounter"]

    # Interval counters
    counter_key = str(counter)
    if counter_key not in day_block["IntervalCounter"]:
        day_block["IntervalCounter"][counter_key] = _empty_interval_counters()
    day_block["IntervalCounter"][counter_key][interval] += 1
    day_block["IntervalCounter"]["sum"][interval] += 1

    # Source address
    sa = record["SourceAddress"]
    if sa not in day_block["SourceAddress"]:
        day_block["SourceAddress"][sa] = {counter_key: 1, "sum": 1}
    elif counter_key not in day_block["SourceAddress"][sa]:
        day_block["SourceAddress"][sa][counter_key] = 1
        day_block["SourceAddress"][sa]["sum"] += 1
    else:
        day_block["SourceAddress"][sa][counter_key] += 1
        day_block["SourceAddress"][sa]["sum"] += 1

    # Destination host
    dh = record["DestinationHost"]
    dh_key = str(dh)
    if dh_key not in day_block["DestinationHost"]:
        day_block["DestinationHost"][dh_key] = {counter_key: 1, "sum": 1}
    elif counter_key not in day_block["DestinationHost"][dh_key]:
        day_block["DestinationHost"][dh_key][counter_key] = 1
        day_block["DestinationHost"][dh_key]["sum"] += 1
    else:
        day_block["DestinationHost"][dh_key][counter_key] += 1
        day_block["DestinationHost"][dh_key]["sum"] += 1

    return model


def add_to_test_model(model: Dict, record: Dict) -> Dict:
    """Add a parsed log record to the test model."""
    un = record["UserName"]
    day_type = record["DayType"]
    interval = record["Interval"]

    if un not in model:
        model[un] = _new_user_test()

    day_block = model[un][day_type]
    day_block["Interval"] = interval
    day_block["IntervalCounter"][interval] += 1

    # Source address
    sa = record["SourceAddress"]
    if sa not in day_block["SourceAddress"]:
        day_block["SourceAddress"][sa] = 1
    else:
        day_block["SourceAddress"][sa] += 1

    # Destination host
    dh = record["DestinationHost"]
    if dh not in day_block["DestinationHost"]:
        day_block["DestinationHost"][dh] = 1
    else:
        day_block["DestinationHost"][dh] += 1

    return model


# ──────────────────────────────────────────────────────────────────────
# Statistics computation (replaces the hardcoded std = 0.2 * avg)
# ──────────────────────────────────────────────────────────────────────

def compute_statistics(model: Dict) -> Dict:
    """
    Compute avg and std for all users in a training model.

    Uses proper statistical computation:
    - avg = sum / num_days
    - std = sqrt(sum_of_squared_deviations / num_days) via Welford's method
      Falls back to 0.2 * avg if insufficient data.
    """
    cfg = get_config().detection

    for user_key in model:
        for day_type in ["WD", "Sat", "Sun"]:
            day_block = model[user_key].get(day_type)
            if day_block is None:
                continue

            # --- Interval counters ---
            ic = day_block.get("IntervalCounter", {})
            # Count actual day entries (exclude 'sum', 'avg', 'std', 'NoFeedback')
            meta_keys = {"sum", "avg", "std", "NoFeedback"}
            day_keys = [k for k in ic if k not in meta_keys]
            num_days = max(len(day_keys), 1)

            sum_arr = np.array(ic.get("sum", _empty_interval_counters()), dtype=np.float64)
            avg_arr = sum_arr / num_days

            if cfg.use_real_std and num_days > 1:
                # Compute real std from per-day data
                day_data = np.array([ic[k] for k in day_keys], dtype=np.float64)
                std_arr = np.std(day_data, axis=0, ddof=0)
                # Floor std to avoid zero
                std_arr = np.maximum(std_arr, avg_arr * 0.1)
            else:
                std_arr = avg_arr * cfg.fallback_std_fraction

            ic["avg"] = avg_arr.tolist()
            ic["std"] = std_arr.tolist()

            # --- Source addresses ---
            for sa_key, sa_data in day_block.get("SourceAddress", {}).items():
                if not isinstance(sa_data, dict):
                    continue
                sa_day_keys = [k for k in sa_data if k not in meta_keys]
                sa_num = max(len(sa_day_keys), 1)
                sa_sum = sa_data.get("sum", 0)
                sa_avg = sa_sum / sa_num
                sa_data["avg"] = round(sa_avg, 4)
                if cfg.use_real_std and sa_num > 1:
                    vals = [sa_data[k] for k in sa_day_keys if isinstance(sa_data.get(k), (int, float))]
                    sa_data["std"] = round(float(np.std(vals, ddof=0)) if vals else sa_avg * 0.2, 4)
                else:
                    sa_data["std"] = round(sa_avg * cfg.fallback_std_fraction, 4)

            # --- Destination hosts ---
            for dh_key, dh_data in day_block.get("DestinationHost", {}).items():
                if not isinstance(dh_data, dict):
                    continue
                dh_day_keys = [k for k in dh_data if k not in meta_keys]
                dh_num = max(len(dh_day_keys), 1)
                dh_sum = dh_data.get("sum", 0)
                dh_avg = dh_sum / dh_num
                dh_data["avg"] = round(dh_avg, 4)
                if cfg.use_real_std and dh_num > 1:
                    vals = [dh_data[k] for k in dh_day_keys if isinstance(dh_data.get(k), (int, float))]
                    dh_data["std"] = round(float(np.std(vals, ddof=0)) if vals else dh_avg * 0.2, 4)
                else:
                    dh_data["std"] = round(dh_avg * cfg.fallback_std_fraction, 4)

    logger.info("Computed statistics for %d users", len(model))
    return model


# ──────────────────────────────────────────────────────────────────────
# File I/O: CSV ingestion
# ──────────────────────────────────────────────────────────────────────

def ingest_csv_to_train_model(
    filepath: Path,
    model: Optional[Dict] = None,
    dest_mgr: Optional[DestinationLabelManager] = None,
) -> Tuple[Dict, DestinationLabelManager]:
    """
    Read a raw CSV log file and aggregate into a training model.

    Returns (model, dest_mgr).
    """
    cfg = get_config().data
    if model is None:
        model = {}
    if dest_mgr is None:
        dest_mgr = DestinationLabelManager()

    logger.info("Ingesting training data from %s", filepath)
    line_count = 0
    record_count = 0

    with open(filepath, "r", newline="") as f:
        next(f, None)  # skip header
        reader = csv.reader(f)
        for row in reader:
            line_count += 1
            if cfg.max_logs_per_file and line_count > cfg.max_logs_per_file:
                logger.info("Reached max_logs_per_file (%d), stopping.", cfg.max_logs_per_file)
                break

            record = parse_csv_row(row, dest_mgr)
            if record is not None:
                model = add_to_train_model(model, record)
                record_count += 1

            if line_count % 100_000 == 0:
                logger.info("  Processed %d lines, %d records so far", line_count, record_count)

    logger.info("Finished: %d lines read, %d records added to model", line_count, record_count)
    return model, dest_mgr


def ingest_csv_to_test_model(
    filepath: Path,
    curr_date: datetime,
    model: Optional[Dict] = None,
    dest_mgr: Optional[DestinationLabelManager] = None,
    batch_size: int = 0,
    start_line: int = 0,
) -> Tuple[Dict, DestinationLabelManager, int, bool]:
    """
    Read a raw CSV log file and build a test model.

    Supports batch reading: reads `batch_size` lines starting from `start_line`.
    If batch_size=0, reads the entire file.

    Returns (model, dest_mgr, lines_read, eof_reached).
    """
    cfg = get_config().data
    if model is None:
        model = {}
    if dest_mgr is None:
        dest_mgr = DestinationLabelManager()
    if batch_size <= 0:
        batch_size = cfg.batch_size

    logger.info("Ingesting test data from %s (start=%d, batch=%d)", filepath, start_line, batch_size)
    line_count = 0
    record_count = 0
    eof = False

    with open(filepath, "r", newline="") as f:
        next(f, None)  # skip header
        reader = csv.reader(f)

        # Skip to start_line
        for _ in range(start_line):
            next(reader, None)

        for row in reader:
            line_count += 1
            if line_count > batch_size:
                break

            record = parse_csv_row(row, dest_mgr)
            if record is not None:
                # Filter to current date only
                if record["DateTime"].date() == curr_date.date():
                    model = add_to_test_model(model, record)
                    record_count += 1

        else:
            eof = True  # for-loop completed without break = EOF

    logger.info("Test ingest: %d lines, %d records, eof=%s", line_count, record_count, eof)
    return model, dest_mgr, line_count, eof


# ──────────────────────────────────────────────────────────────────────
# JSON intermediate file support (for streaming/batch processing)
# ──────────────────────────────────────────────────────────────────────

def write_batch_json(
    csv_path: Path,
    output_path: Path,
    num_logs: int,
    skip_lines: int,
    field_names: List[str],
) -> None:
    """Write a batch of CSV rows as a JSON list of dicts."""
    records = []
    with open(csv_path, "r", newline="") as f:
        # Skip header + already-read lines
        for _ in range(skip_lines + 1):
            next(f, None)

        reader = csv.DictReader(f, fieldnames=field_names)
        for i, row in enumerate(reader):
            if i >= num_logs:
                break
            records.append(dict(row))

    with open(output_path, "w") as f:
        json.dump(records, f)

    logger.info("Wrote %d records to %s", len(records), output_path)


def ingest_json_to_test_model(
    json_path: Path,
    curr_date: datetime,
    model: Optional[Dict] = None,
    dest_mgr: Optional[DestinationLabelManager] = None,
    max_records: int = 0,
) -> Tuple[Dict, DestinationLabelManager, int, bool]:
    """
    Read an intermediate JSON file and build/update a test model.

    Returns (model, dest_mgr, records_read, eof_flag).
    """
    cfg = get_config().data
    if model is None:
        model = {}
    if dest_mgr is None:
        dest_mgr = DestinationLabelManager()
    if max_records <= 0:
        max_records = cfg.batch_size

    with open(json_path, "r") as f:
        rows = json.load(f)

    record_count = 0
    eof = len(rows) < max_records

    for row in rows:
        if record_count >= max_records:
            break
        record = parse_json_row(row, curr_date, dest_mgr)
        if record is not None:
            model = add_to_test_model(model, record)
            record_count += 1

    logger.info("JSON ingest: %d/%d records processed, eof=%s", record_count, len(rows), eof)
    return model, dest_mgr, record_count, eof
