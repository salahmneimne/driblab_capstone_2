"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 4: Fill event labels into existing columns
=============================================================

Takes each *_training_data.csv (output of Step 3) and fills the two
existing label columns for every row that currently has them as NaN:

  event_type_id   (int, from dim_event_type)
  event_type_name (str, from dim_event_type)

No new columns are added. The table shape stays identical.

Filling rules
-------------
1. ROWS WITH has_event=True AND event in TARGET_EVENTS (12 classes)
   Already filled from Step 3. Left untouched.

2. ROWS WITH has_event=True AND event NOT in TARGET_EVENTS
   Collapsed to NO EVENT (event_type_id=0, event_type_name="NO EVENT").
   These are administrative or kinematically undetectable events (FOUL,
   BALL TOUCH, TAKEON, SUBSTITUTION, CARD, etc.) confirmed excluded by
   the EDA. Keeping them as-is would teach the model to detect things it
   cannot detect from tracking data alone.

3. ROWS WITHIN ±3 FRAMES (300 ms) OF A TARGET EVENT ANCHOR
   event_type_id   = nearest target event anchor's event_type_id
   event_type_name = nearest target event anchor's event_type_name
   Non-target events do not create a suppression zone — their surrounding
   frames correctly remain NO EVENT.
   Period boundary respected: no propagation across half-time.

4. ALL OTHER ROWS
   event_type_id   = 0         ("NO EVENT" in dim_event_type)
   event_type_name = "NO EVENT"

Target event classes (12, confirmed in EDA Section 1e)
------------------------------------------------------
PASS, TACKLE, INTERCEPTION, CLEARANCE, BALL RECOVERY,
AERIAL, SAVE, MISSED SHOT, SAVED SHOT, GOAL,
DISPOSSESSED, KEEPER PICKUP

Excluded (collapsed to NO EVENT)
---------------------------------
FOUL          — no contact signal from (x,y) dots; needs pose/video
BALL TOUCH    — too fine-grained for 10 Hz; indistinguishable from carry
TAKEON        — dribble attempt; indistinguishable from carry at 10 Hz
CHALLENGE     — broad overlap with TACKLE; causes label noise
CARD          — administrative consequence of FOUL
SUBSTITUTION  — administrative; no ball/position signal
END           — match boundary marker
FORMATION CHANGE — tactical annotation; no event-level tracking signal
CLAIM         — goalkeeper sub-event; covered by SAVE
OFFSIDE PASS  — requires offside line computation; out of scope
CHANCE MISSED — editorial label; not independently detectable

Binary classifier note (mentor instruction)
-------------------------------------------
For the first model, pick one target class (e.g. PASS) and treat all
other rows — including the other 11 target classes — as the negative
class. The clean event_type_name column here supports that directly:
    df['target'] = (df['event_type_name'] == 'PASS').astype(int)

Usage
-----
  python 04_label_windows.py --joined ./joined --out ./labelled
  python 04_label_windows.py --joined ./joined --match 745399 --out ./labelled
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HALF_WIN     = 3           # frames either side = 300 ms at 10 fps
BG_TYPE_ID   = 0
BG_TYPE_NAME = "NO EVENT"

# Confirmed in EDA Section 1e — do not modify without revisiting the EDA
TARGET_EVENTS = {
    "PASS",
    "TACKLE",
    "INTERCEPTION",
    "CLEARANCE",
    "BALL RECOVERY",
    "AERIAL",
    "SAVE",
    "MISSED SHOT",
    "SAVED SHOT",
    "GOAL",
    "DISPOSSESSED",
    "KEEPER PICKUP",
}


def fill_labels(df: pd.DataFrame, half_win: int) -> pd.DataFrame:
    """
    Fill event_type_id and event_type_name for every row using the
    rules described in the module docstring.
    """
    df = df.copy()

    # ── Step 1: collapse non-target event rows to NO EVENT ───────────────────
    non_target_mask = (
        df["has_event"].astype(bool) &
        (~df["event_type_name"].isin(TARGET_EVENTS))
    )
    df.loc[non_target_mask, "event_type_id"]   = float(BG_TYPE_ID)
    df.loc[non_target_mask, "event_type_name"] = BG_TYPE_NAME

    # Null out event-instance columns for relabelled rows.
    # Keeping player_id, team_id, x_m etc. from a FOUL or SUBSTITUTION
    # while calling the row NO EVENT would give the model contradictory
    # signals — it could learn to associate those player/location values
    # with background, which would corrupt the feature engineering step.
    # has_event stays True so anchor detection in Step 5 still works.
    EVENT_INSTANCE_COLS = [
        "event_id",
        "team_id", "team_name", "player_id", "player_name",
        "period_id", "min", "sec", "milisec",
        "x", "y", "x_end", "y_end",
        "outcome", "event_abs_sec", "attack_dir",
        "x_m", "y_m", "x_end_m", "y_end_m",
        "time_residual_s", "time_matched",
    ]
    cols_to_null = [c for c in EVENT_INSTANCE_COLS if c in df.columns]
    df.loc[non_target_mask, cols_to_null] = np.nan

    # ── Step 2: build anchor arrays from TARGET event rows only ──────────────
    target_ev     = df[df["has_event"].astype(bool) &
                       df["event_type_name"].isin(TARGET_EVENTS)]
    ev_frames     = target_ev["frame"].values.astype(int)
    ev_type_ids   = target_ev["event_type_id"].values
    ev_type_names = target_ev["event_type_name"].values
    ev_periods    = target_ev["period"].values.astype(int)

    # ── Step 3: fill non-event rows ──────────────────────────────────────────
    needs_fill   = ~df["has_event"].astype(bool)
    fill_idx     = df.index[needs_fill]
    fill_frames  = df.loc[fill_idx, "frame"].values.astype(int)
    fill_periods = df.loc[fill_idx, "period"].values.astype(int)

    new_type_ids   = np.full(len(fill_idx), float(BG_TYPE_ID))
    new_type_names = np.full(len(fill_idx), BG_TYPE_NAME, dtype=object)

    for i, (f, fp) in enumerate(zip(fill_frames, fill_periods)):
        same_period = ev_periods == fp
        dists       = np.abs(ev_frames - f)
        mask        = same_period & (dists <= half_win)

        if mask.any():
            candidates  = np.where(mask)[0]
            nearest_idx = candidates[np.argmin(dists[candidates])]
            new_type_ids[i]   = ev_type_ids[nearest_idx]
            new_type_names[i] = ev_type_names[nearest_idx]

    df.loc[fill_idx, "event_type_id"]   = new_type_ids
    df.loc[fill_idx, "event_type_name"] = new_type_names

    return df


def print_stats(df: pd.DataFrame, match_id: str) -> None:
    counts = df["event_type_name"].value_counts()
    bg_n   = counts.get(BG_TYPE_NAME, 0)
    ev_n   = len(df) - bg_n
    print(f"\n[{match_id}]  {len(df)} rows  "
          f"({ev_n} event-context / {bg_n} background)")
    print(counts.to_string())


def run_checks(df: pd.DataFrame, half_win: int) -> None:
    # 1. No nulls in either label column
    assert df["event_type_id"].notna().all(),   "event_type_id has NaN after fill"
    assert df["event_type_name"].notna().all(), "event_type_name has NaN after fill"

    # 2. No label outside TARGET_EVENTS + NO EVENT
    allowed = TARGET_EVENTS | {BG_TYPE_NAME}
    unexpected = set(df["event_type_name"].unique()) - allowed
    assert not unexpected, f"Unexpected labels found: {unexpected}"

    # 3. No row within ±half_win frames of a same-period TARGET event is NO EVENT
    forbidden_by_period: dict = {}
    target_rows = df[
        df["has_event"].astype(bool) &
        df["event_type_name"].isin(TARGET_EVENTS)
    ]
    for _, row in target_rows.iterrows():
        p  = int(row["period"])
        ef = int(row["frame"])
        if p not in forbidden_by_period:
            forbidden_by_period[p] = set()
        forbidden_by_period[p].update(range(ef - half_win, ef + half_win + 1))

    # Only check has_event=False rows — collapsed non-target event rows
    # (has_event=True) are intentionally NO EVENT and live at event frames,
    # so they will legitimately appear inside the forbidden zone.
    bg_rows = df[
        (df["event_type_name"] == BG_TYPE_NAME) &
        (~df["has_event"].astype(bool))
    ]
    bad_frames = [
        int(r["frame"])
        for _, r in bg_rows.iterrows()
        if int(r["frame"]) in forbidden_by_period.get(int(r["period"]), set())
    ]
    assert len(bad_frames) == 0, (
        f"{len(bad_frames)} rows within same-period target event window "
        f"labelled NO EVENT: {bad_frames[:5]}"
    )

    print("  [OK] sanity checks passed")


def main():
    ap = argparse.ArgumentParser(
        description="Capstone Step 4 — fill event labels into existing columns"
    )
    ap.add_argument("--joined",   default="./results/joined",
                    help="Directory with *_training_data.csv files from Step 3")
    ap.add_argument("--match",    default=None,
                    help="Single match id (default: all matches in --joined)")
    ap.add_argument("--out",      default="./results/labelled",
                    help="Output directory for *_labelled.csv files")
    ap.add_argument("--half-win", type=int, default=HALF_WIN,
                    help=f"Half-window in frames (default {HALF_WIN} = 300 ms)")
    args = ap.parse_args()

    half_win   = args.half_win
    joined_dir = Path(args.joined)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match:
        match_ids = [args.match]
    else:
        match_ids = sorted({
            p.name.split("_")[0]
            for p in joined_dir.glob("*_training_data.csv")
        })

    if not match_ids:
        print(f"No *_training_data.csv files found in {joined_dir}")
        return

    for match_id in match_ids:
        src = joined_dir / f"{match_id}_training_data.csv"
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue

        df     = pd.read_csv(src, low_memory=False)
        filled = fill_labels(df, half_win)
        print_stats(filled, match_id)
        run_checks(filled, half_win)

        out_path = out_dir / f"{match_id}_labelled.csv"
        filled.to_csv(out_path, index=False)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
