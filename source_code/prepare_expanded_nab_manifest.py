"""Download and pre-screen additional real NAB files with complete held-out test events.

Run from source_code/: python prepare_expanded_nab_manifest.py --additional 15
This uses the NAB labels file already present in data/nab/labels/combined_windows.json.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "nab"
LABELS = DATA_ROOT / "labels" / "combined_windows.json"
RAW = "https://raw.githubusercontent.com/numenta/NAB/master"
DOMAIN_BY_PREFIX = {
    "realAWSCloudwatch/": "AWS CloudWatch",
    "realAdExchange/": "Ad Exchange",
    "realKnownCause/": "Known Cause",
    "realTraffic/": "Traffic",
    "realTweets/": "Twitter",
}


def load_series(relative: str) -> pd.DataFrame:
    path = DATA_ROOT / "data" / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(f"{RAW}/data/{relative}", timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    frame = pd.read_csv(path)[["timestamp", "value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp", keep="last")
    frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan).ffill()
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def eligible(relative: str, events: list, min_rows: int = 600) -> tuple[bool, dict]:
    frame = load_series(relative)
    n = len(frame)
    if n < min_rows:
        return False, {"rows": n, "reason": "too_short_for_W100"}
    test_start_idx = int(n * 0.80)
    test_start = frame.timestamp.iloc[test_start_idx]
    test_end = frame.timestamp.iloc[-1]
    ts = set(pd.DatetimeIndex(frame.timestamp.iloc[test_start_idx:]))
    complete = []
    positive_test_points = 0
    for start_s, end_s in events:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        if test_start <= start <= end <= test_end and start in ts and end in ts:
            complete.append((start_s, end_s))
            positive_test_points += int(((frame.timestamp.iloc[test_start_idx:] >= start) & (frame.timestamp.iloc[test_start_idx:] <= end)).sum())
    train_end = int(n * 0.60)
    # Require a minimally useful normal-only W=100 training set.
    labels_train = np.zeros(train_end, dtype=np.int8)
    for start_s, end_s in events:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        labels_train |= ((frame.timestamp.iloc[:train_end] >= start) & (frame.timestamp.iloc[:train_end] <= end)).to_numpy(dtype=np.int8)
    prefix = np.concatenate([[0], np.cumsum(labels_train)])
    ends = np.arange(99, train_end)
    clean = int(np.sum((prefix[ends + 1] - prefix[ends - 99]) == 0)) if len(ends) else 0
    val_end = int(n * 0.80)
    val_ends = np.arange(train_end, val_end)
    labels_all = np.zeros(n, dtype=np.int8)
    for start_s, end_s in events:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        labels_all |= ((frame.timestamp >= start) & (frame.timestamp <= end)).to_numpy(dtype=np.int8)
    prefix_all = np.concatenate([[0], np.cumsum(labels_all)])
    val_starts = val_ends - 99
    clean_val = int(np.sum((prefix_all[val_ends + 1] - prefix_all[val_starts]) == 0)) if len(val_ends) else 0
    ok = bool(complete and positive_test_points > 0 and clean >= 100 and clean_val >= 50)
    return ok, {
        "rows": n, "complete_test_events": len(complete),
        "positive_test_points": positive_test_points,
        "normal_W100_training_windows": clean,
        "normal_W100_validation_windows_for_early_stopping": clean_val,
        "reason": "eligible" if ok else "no_complete_test_event_or_insufficient_clean_train_validation_windows",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--additional", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not LABELS.exists():
        raise SystemExit(f"Missing NAB labels file: {LABELS}")
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    selected = dict(base["selected_files"])
    existing = set(selected)
    pools = {domain: [] for domain in DOMAIN_BY_PREFIX.values()}
    for relative, events in labels.items():
        if relative in existing or not events:
            continue
        for prefix, domain in DOMAIN_BY_PREFIX.items():
            if relative.startswith(prefix):
                pools[domain].append((relative, events))
                break
    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)

    # Balance additions across real NAB domains; do not add synthetic/artificial series.
    domain_order = list(pools)
    additions, audit = [], []
    while len(additions) < args.additional:
        progressed = False
        for domain in domain_order:
            if len(additions) >= args.additional:
                break
            already = sum(1 for _, d in additions if d == domain)
            if already >= 3:
                continue
            while pools[domain]:
                relative, events = pools[domain].pop()
                try:
                    ok, detail = eligible(relative, events)
                except Exception as exc:
                    ok, detail = False, {"reason": f"download_or_parse_error: {exc}"}
                audit.append({"dataset": relative, "domain": domain, **detail})
                if ok:
                    additions.append((relative, domain))
                    print(f"SELECTED {relative}: {detail}")
                    progressed = True
                    break
                print(f"SKIPPED {relative}: {detail}")
        if not progressed:
            break

    # If one domain has fewer eligible files, fill from remaining real-domain pools.
    if len(additions) < args.additional:
        for domain in domain_order:
            while pools[domain] and len(additions) < args.additional:
                relative, events = pools[domain].pop()
                try:
                    ok, detail = eligible(relative, events)
                except Exception as exc:
                    ok, detail = False, {"reason": f"download_or_parse_error: {exc}"}
                audit.append({"dataset": relative, "domain": domain, **detail})
                if ok:
                    additions.append((relative, domain))
                    print(f"SELECTED {relative}: {detail}")

    for relative, domain in additions:
        selected[relative] = domain
    cfg = json.loads((ROOT / "config_v2.json").read_text(encoding="utf-8"))
    cfg["selected_files"] = selected
    (ROOT / "config_v2.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    report = {
        "requested_additional": args.additional,
        "selected_additional": len(additions),
        "total_selected": len(selected),
        "selected_files": [{"dataset": p, "domain": d} for p, d in additions],
        "screening_audit": audit,
        "eligibility": "At least one complete labeled event wholly inside the 20% chronological test partition, positive test labels, >=100 normal-only W=100 training windows.",
    }
    (ROOT / "expanded_dataset_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSelected {len(additions)} additional files; total={len(selected)}.")
    print(f"Updated: {ROOT / 'config_v2.json'}")
    if len(additions) < args.additional:
        print("WARNING: NAB data did not provide enough eligible files under the screening rules. Do not lower criteria silently; inspect expanded_dataset_selection.json.")

if __name__ == "__main__":
    main()
