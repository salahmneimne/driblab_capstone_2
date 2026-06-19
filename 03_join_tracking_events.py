"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 3: Join tracking + events
=============================================================

Combines the outputs of 02_align_and_clean.py into one file per match:
every tracking frame, left-joined with any event(s) matched to it via
`matched_frame` (computed in Step 2's sync_event_time).

- Frames with no matching event get NaN event columns (has_event=False).
- Frames with multiple matching events (e.g. paired AERIAL duels) appear
  as multiple rows, one per event, with the same frame-level columns
  repeated (has_event=True for all of them).

Diagnostic-only columns from Step 2 (matched_frame_idx, the per-period
offset calibration columns) are dropped here; `time_residual_s` /
`time_matched` are kept so events that couldn't be matched within
tolerance remain visible.

Usage
-----
  python 03_join_tracking_events.py --aligned ./aligned --match 678949 --out ./joined
  python 03_join_tracking_events.py --aligned ./aligned --out ./joined   # all matches in --aligned
"""

import argparse
from pathlib import Path

import pandas as pd

EVENT_DIAGNOSTIC_COLS = [
    "matched_frame_idx",
    "period_clock_offset_s",
    "period_offset_spatial_dist_m",
    "period_offset_n_anchors",
]


def join_match(aligned_dir: Path, match_id: str) -> pd.DataFrame:
    frames = pd.read_csv(aligned_dir / f"{match_id}_tracking_clean.csv", low_memory=False)
    events = pd.read_csv(aligned_dir / f"{match_id}_events_aligned.csv")

    events = events.drop(columns=EVENT_DIAGNOSTIC_COLS)
    events = events.rename(columns={"matched_frame": "frame"})

    joined = frames.merge(events, on="frame", how="left")
    # defragment after wide merge before adding columns
    joined = joined.copy()
    joined["has_event"] = joined["event_id"].notna()
    return joined


def main():
    ap = argparse.ArgumentParser(description="Capstone Step 3 — join tracking + events")
    ap.add_argument("--aligned", default="./results/aligned")
    ap.add_argument("--match", help="match id, e.g. 678949 (default: all matches found in --aligned)")
    ap.add_argument("--out", default="./results/joined")
    args = ap.parse_args()

    aligned_dir, out_dir = Path(args.aligned), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match:
        match_ids = [args.match]
    else:
        match_ids = sorted({
            p.name.split("_")[0] for p in aligned_dir.glob("*_tracking_clean.csv")
        })

    for match_id in match_ids:
        joined = join_match(aligned_dir, match_id)
        out_path = out_dir / f"{match_id}_training_data.csv"
        joined.to_csv(out_path, index=False)

        n_frames = joined["frame"].nunique()
        n_events = int(joined["has_event"].sum())
        event_rows = joined.loc[joined["has_event"]]
        n_unmatched = int((~event_rows["time_matched"].astype(bool)).sum())
        print(f"{match_id}: {n_frames} frames, {len(joined)} rows, "
              f"{n_events} event rows ({n_unmatched} outside tolerance) -> {out_path}")


if __name__ == "__main__":
    main()
