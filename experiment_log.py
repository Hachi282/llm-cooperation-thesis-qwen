"""JSONL logger — schema-compatible with donor_game/logs/.

Paper saves one big JSON to Google Drive at the end. We append per
event during the run so:
  * the parent donor_game/ analysis scripts can read it (same schema);
  * a crash mid-run still leaves usable partial data;
  * progress can be tailed live.

Two event types match donor_game/:
  type="strategy" : ts, type, agent_name, generation, model, had_inherited,
                    prompt, raw_output, justification, strategy
  type="donation" : ts, type, generation, game, round, donor_name,
                    recipient_name, donor_resources_before,
                    recipient_resources_before, donation_amount,
                    donation_pct, donor_resources_after,
                    recipient_resources_after, donor_reputation_after,
                    justification, raw_response
"""
import json
import os
import threading
from datetime import datetime

_LOG_FILE = None
_LOG_PATH = None
_LOCK = threading.Lock()  # paper runs LLM calls in parallel → log writes must be thread-safe


def set_log_path(path):
    """Open `path` for appending. Creates parent dirs if needed."""
    global _LOG_FILE, _LOG_PATH
    close_log()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _LOG_FILE = open(path, "a", encoding="utf-8")
    _LOG_PATH = path


def log_event(**fields):
    """Write one JSON record. Silently no-ops if no log path set."""
    if _LOG_FILE is None:
        return
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
    with _LOCK:
        _LOG_FILE.write(json.dumps(record, ensure_ascii=False) + "\n")
        _LOG_FILE.flush()


def close_log():
    global _LOG_FILE, _LOG_PATH
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None
        _LOG_PATH = None


def current_log_path():
    return _LOG_PATH
