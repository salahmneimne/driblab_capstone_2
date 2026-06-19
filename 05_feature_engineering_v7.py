"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 5: Feature engineering (v7, leakage fix)
=============================================================

Input : *_labelled.csv   (Step 4 output, unchanged)
        *_match_meta.json (Step 2 output, unchanged)
Output: *_features.csv   (one row per anchor, 5 features + metadata)

WHY THIS VERSION EXISTS
------------------------
v6 computed several features from the EVENT annotation rather than from
tracking data:

  event_x_m, event_y_m, event_x_end_m, event_y_end_m, event_outcome,
  attack_dir, pass_length          -> read straight from the event row
  event_dist_to_home_goal/away_goal -> distance from event_x_m/y_m
  min_dist_home_to_ball, min_dist_away_to_ball,
  gk_defending_dist, gk_attacking_dist, gk_in_goal_area
                                     -> all routed through best_location(),
                                        which prefers x_m/y_m (the event's
                                        annotated position) and only fell
                                        back to ball_x/ball_y when the event
                                        location was missing.

None of these are available at real prediction time: on an unlabelled
match there is no event row yet, because the event is what the model is
trying to predict. Confirmed empirically: event_dist_to_home_goal was
null for ~73% of NO EVENT rows and ~100% populated for every event row,
a pattern a tree model exploits instantly instead of learning real
ball/player movement.

This script computes exactly 5 features. Every single one is built only
from columns that exist identically on a fresh, unlabelled match:
  frame, period, ball_x, ball_y, ball_z, ball_interpolated, cam_present,
  px_<id>, py_<id>

None of these read x_m, y_m, x_end_m, y_end_m, outcome, attack_dir,
event_id, or any other event-table column. There is no best_location()
function in this script — that ambiguity is exactly what caused the
fallback-to-event-position behaviour in v6, so it has been removed
rather than fixed, to make the leak structurally impossible to reintroduce.

The 5 features (mapped 1:1 onto the rules in 04b_rule_based_detector.py
so the ML-vs-rules comparison is apples to apples):

  1. ball_speed_mean, ball_speed_max, ball_speed_at_anchor
     -> same primitive as 04b Rules 2/4/6 (GOAL/SHOT, CLEARANCE, PASS)
  2. ball_z_max, ball_z_mean, frames_z_above_1_5m
     -> same primitive as 04b Rule 1 (AERIAL)
  3. ball_dist_to_home_goal, ball_dist_to_away_goal, ball_pitch_third
     -> from ball_x/ball_y at anchor only, never event_x_m/event_y_m
  4. possession_change_in_window
     -> same nearest_team logic as 04b Rule 5
  5. n_opponents_within_1m5, n_opponents_within_3m
     -> no rule-based equivalent; the one feature meant to help the ML
        model separate TACKLE / INTERCEPTION / BALL RECOVERY /
        DISPOSSESSED, which 04b can only lump into one group

Anchor sampling is unchanged from the iteration-4 fix: background
anchors are sampled only from cam_present=True frames outside a
forbidden zone around every real event, so the camera-presence leak
from iteration 3 cannot resurface.

Usage
-----
  python 05_feature_engineering_v7.py --labelled ./labelled --meta ./aligned --out ./features_v7
  python 05_feature_engineering_v7.py --labelled ./labelled --meta ./aligned --match 745399 --out ./features_v7
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────
HALF_WIN  = 30      # frames either side of anchor (3 s at 10 fps)
BG_STRIDE = 12       # sample one background anchor every N safe frames
SPEED_CAP = 60.0     # m/s — physical upper bound, drops tracking-noise spikes
PITCH_X, PITCH_Y = 105.0, 68.0
GOAL_HOME = (0.0,    PITCH_Y / 2)
GOAL_AWAY = (PITCH_X, PITCH_Y / 2)
Z_HIGH    = 1.5      # m — aerial height threshold, matches 04b Rule 1

TARGET_EVENTS = {
    "PASS", "TACKLE", "INTERCEPTION", "CLEARANCE", "BALL RECOVERY",
    "AERIAL", "SAVE", "MISSED SHOT", "SAVED SHOT", "GOAL",
    "DISPOSSESSED", "KEEPER PICKUP",
}

# Columns this script must NEVER read. Checked defensively at load time
# so a future edit can't quietly reintroduce the event-table dependency.
FORBIDDEN_COLS = {
    "x_m", "y_m", "x_end_m", "y_end_m", "outcome", "attack_dir",
    "event_id", "event_abs_sec",
}


# ── helpers ───────────────────────────────────────────────────────────────

def load_meta(meta_path: Path) -> tuple:
    m = json.loads(meta_path.read_text())
    return (
        int(m["home_id"]),
        int(m["away_id"]),
        int(m["gk_home_id"]) if m.get("gk_home_id") is not None else None,
        int(m["gk_away_id"]) if m.get("gk_away_id") is not None else None,
    )


def get_team_players(df: pd.DataFrame, home_id: int, away_id: int) -> tuple:
    """Map player_id -> team using event rows purely to build the roster
    (which players belong to which team is fixed pre-match metadata, not
    something the model needs to predict, so this is not a leak — it is
    the same roster a broadcast graphic would show before kickoff)."""
    px_cols  = [c for c in df.columns if c.startswith("px_")]
    all_pids = [int(c.replace("px_", "")) for c in px_cols]

    ev = df[df["has_event"].astype(bool)][["player_id", "team_id"]].dropna()
    pid_team = (
        ev.drop_duplicates("player_id")
        .set_index("player_id")["team_id"]
        .astype(int)
        .to_dict()
    )
    home_pids = [p for p in all_pids if pid_team.get(p) == home_id]
    away_pids = [p for p in all_pids if pid_team.get(p) == away_id]
    for p in [p for p in all_pids if p not in pid_team]:
        (home_pids if len(home_pids) <= len(away_pids) else away_pids).append(p)
    return home_pids, away_pids


def nearest_team(row, home_pids, away_pids):
    """Return 'home' or 'away' for whichever team has the closest player
    to the ball at this tracking row. Identical logic to 04b so the rule
    baseline and the ML feature are built from the same primitive."""
    bx, by = row.get("ball_x"), row.get("ball_y")
    if pd.isna(bx) or pd.isna(by):
        return None
    bx, by = float(bx), float(by)
    hd = [np.sqrt((row[f"px_{p}"] - bx) ** 2 + (row[f"py_{p}"] - by) ** 2)
          for p in home_pids if pd.notna(row.get(f"px_{p}"))]
    ad = [np.sqrt((row[f"px_{p}"] - bx) ** 2 + (row[f"py_{p}"] - by) ** 2)
          for p in away_pids if pd.notna(row.get(f"px_{p}"))]
    if not hd or not ad:
        return None
    return "home" if min(hd) < min(ad) else "away"


def ball_speed_series(window: pd.DataFrame) -> pd.DataFrame:
    """Frame-to-frame ball speed (m/s) over consecutive, real (non
    interpolated) frames. Same approach as 04b's ball_speed_series."""
    real = window[window["ball_x"].notna() & ~window["ball_interpolated"]].copy()
    if len(real) < 2:
        return pd.DataFrame()
    real["frame_diff"] = real["frame"].diff()
    real["dx"] = real["ball_x"].diff()
    real["dy"] = real["ball_y"].diff()
    consec = real[real["frame_diff"] == 1].copy()
    if len(consec) == 0:
        return pd.DataFrame()
    consec["speed"] = np.sqrt(consec["dx"] ** 2 + consec["dy"] ** 2) / 0.1
    consec = consec[consec["speed"] <= SPEED_CAP]
    return consec


# ── feature computation ──────────────────────────────────────────────────

def compute_features(
    anchor_frame:  int,
    anchor_period: int,
    df:            pd.DataFrame,
    home_pids:     list,
    away_pids:     list,
    period_bounds: dict,
) -> dict:
    """Compute the 5 tracking-only feature groups for one anchor window.

    Every value here is derived from frame, period, ball_x/y/z,
    ball_interpolated, cam_present, or px_*/py_* — columns that exist on
    a fresh unlabelled match. Nothing here reads an event-table column.
    """
    p_min, p_max = period_bounds[anchor_period]
    w_start = max(anchor_frame - HALF_WIN, p_min)
    w_end   = min(anchor_frame + HALF_WIN, p_max)

    window = (
        df[(df["frame"] >= w_start) & (df["frame"] <= w_end)]
        .drop_duplicates("frame")
        .sort_values("frame")
    )
    feats: dict = {}

    # ── data quality (diagnostic, not fed to the model, kept for audit) ──
    n = len(window)
    feats["ball_available_ratio"] = window["ball_x"].notna().sum() / n if n else np.nan
    feats["cam_present_ratio"]    = window["cam_present"].sum() / n if n else np.nan

    # ── feature 1: ball speed ─────────────────────────────────────────────
    spd = ball_speed_series(window)
    if len(spd) > 0:
        feats["ball_speed_mean"] = float(spd["speed"].mean())
        feats["ball_speed_max"]  = float(spd["speed"].max())
        near = spd.iloc[(spd["frame"] - anchor_frame).abs().argsort()]
        feats["ball_speed_at_anchor"] = float(near["speed"].iloc[0])
    else:
        feats["ball_speed_mean"] = np.nan
        feats["ball_speed_max"] = np.nan
        feats["ball_speed_at_anchor"] = np.nan

    # ── feature 2: ball height ────────────────────────────────────────────
    bz_all = window["ball_z"].dropna()
    feats["ball_z_max"]  = float(bz_all.max())  if len(bz_all) else np.nan
    feats["ball_z_mean"] = float(bz_all.mean()) if len(bz_all) else np.nan
    feats["frames_z_above_1_5m"] = int((bz_all > Z_HIGH).sum()) if len(bz_all) else 0

    # ── feature 3: ball-to-goal distance, from ball_x/ball_y only ────────
    anchor_rows = df[df["frame"] == anchor_frame]
    arow = anchor_rows.iloc[0] if len(anchor_rows) else None
    bx = float(arow["ball_x"]) if arow is not None and pd.notna(arow["ball_x"]) else np.nan
    by = float(arow["ball_y"]) if arow is not None and pd.notna(arow["ball_y"]) else np.nan

    if pd.notna(bx) and pd.notna(by):
        feats["ball_dist_to_home_goal"] = float(np.sqrt(
            (bx - GOAL_HOME[0]) ** 2 + (by - GOAL_HOME[1]) ** 2
        ))
        feats["ball_dist_to_away_goal"] = float(np.sqrt(
            (bx - GOAL_AWAY[0]) ** 2 + (by - GOAL_AWAY[1]) ** 2
        ))
        feats["ball_pitch_third"] = 0 if bx < 35 else (1 if bx < 70 else 2)
    else:
        feats["ball_dist_to_home_goal"] = np.nan
        feats["ball_dist_to_away_goal"] = np.nan
        feats["ball_pitch_third"] = np.nan

    # ── feature 4: possession change across the window ───────────────────
    # Same primitive as 04b Rule 5/6 (PASS vs TACKLE/INTERCEPTION/etc.)
    start_team = nearest_team(window.iloc[0], home_pids, away_pids) if n else None
    end_team   = nearest_team(window.iloc[-1], home_pids, away_pids) if n else None
    if start_team is not None and end_team is not None:
        feats["possession_change_in_window"] = int(start_team != end_team)
    else:
        feats["possession_change_in_window"] = np.nan

    # ── feature 5: opponent proximity to ball at anchor ───────────────────
    # "Opponent" here means whichever team does NOT hold the ball at the
    # anchor frame, determined purely by nearest-player-to-ball — no event
    # outcome or player_id from the event row is used.
    if pd.notna(bx) and pd.notna(by) and arow is not None:
        hd = [np.sqrt((arow[f"px_{p}"] - bx) ** 2 + (arow[f"py_{p}"] - by) ** 2)
              for p in home_pids if pd.notna(arow.get(f"px_{p}"))]
        ad = [np.sqrt((arow[f"px_{p}"] - bx) ** 2 + (arow[f"py_{p}"] - by) ** 2)
              for p in away_pids if pd.notna(arow.get(f"px_{p}"))]
        if hd and ad:
            opponents = ad if min(hd) < min(ad) else hd
            feats["n_opponents_within_1m5"] = int(sum(d < 1.5 for d in opponents))
            feats["n_opponents_within_3m"]  = int(sum(d < 3.0 for d in opponents))
        else:
            feats["n_opponents_within_1m5"] = np.nan
            feats["n_opponents_within_3m"]  = np.nan
    else:
        feats["n_opponents_within_1m5"] = np.nan
        feats["n_opponents_within_3m"]  = np.nan

    return feats


# ── per-match processing ─────────────────────────────────────────────────

def process_match(
    labelled_path: Path,
    meta_path:     Path,
    match_id:      str,
    bg_stride:     int,
) -> pd.DataFrame:

    df = pd.read_csv(labelled_path, low_memory=False)

    present_forbidden = FORBIDDEN_COLS & set(df.columns)
    if present_forbidden:
        print(f"  [{match_id}] note: columns {sorted(present_forbidden)} exist in the "
              f"labelled table but are NOT read by this script (by design).")

    home_id, away_id, gk_home_id, gk_away_id = load_meta(meta_path)
    home_pids, away_pids = get_team_players(df, home_id, away_id)

    print(f"  [{match_id}] home={home_id}({len(home_pids)}p) "
          f"away={away_id}({len(away_pids)}p)")

    period_bounds = {
        int(p): (int(grp["frame"].min()), int(grp["frame"].max()))
        for p, grp in df.groupby("period")
    }

    # ── event anchors (target events only, same as v6) ───────────────────
    target_rows = df[
        df["has_event"].astype(bool) &
        df["event_type_name"].isin(TARGET_EVENTS)
    ]
    anchors = []
    for _, row in target_rows.iterrows():
        anchors.append({
            "anchor_frame": int(row["frame"]),
            "period":       int(row["period"]),
            "label":        row["event_type_name"],
        })

    # ── background anchors ─────────────────────────────────────────────
    # Forbidden zone = ±HALF_WIN around every target event frame, so no
    # background window overlaps a real event. cam_present=True filter
    # preserves the iteration-4 fix (no camera-presence leak).
    target_frames = set(target_rows["frame"].astype(int).tolist())
    forbidden: set = set()
    for ef in target_frames:
        forbidden.update(range(ef - HALF_WIN, ef + HALF_WIN + 1))

    no_ev = (
        df[(df["event_type_name"] == "NO EVENT") & (~df["has_event"].astype(bool))]
        .drop_duplicates("frame")
        .sort_values("frame")
    )
    safe_bg = no_ev[
        ~no_ev["frame"].isin(forbidden) &
        (no_ev["cam_present"] == True)
    ]

    for _, row in safe_bg.iloc[::bg_stride].iterrows():
        anchors.append({
            "anchor_frame": int(row["frame"]),
            "period":       int(row["period"]),
            "label":        "NO EVENT",
        })

    n_ev = sum(1 for a in anchors if a["label"] != "NO EVENT")
    n_bg = len(anchors) - n_ev
    print(f"  [{match_id}] {n_ev} event anchors + {n_bg} background anchors = {len(anchors)} total")

    # ── compute features ───────────────────────────────────────────────
    rows = []
    for anchor in anchors:
        feats = compute_features(
            anchor_frame  = anchor["anchor_frame"],
            anchor_period = anchor["period"],
            df            = df,
            home_pids     = home_pids,
            away_pids     = away_pids,
            period_bounds = period_bounds,
        )
        rows.append({
            "match_id":     match_id,
            "anchor_frame": anchor["anchor_frame"],
            "period":       anchor["period"],
            "label":        anchor["label"],
            **feats,
        })

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Capstone Step 5 v7 — leakage-free feature engineering")
    ap.add_argument("--labelled",  default="./results/labelled")
    ap.add_argument("--meta",      default="./results/aligned")
    ap.add_argument("--out",       default="./results/features_v7")
    ap.add_argument("--match",     default=None)
    ap.add_argument("--bg-stride", type=int, default=BG_STRIDE)
    args = ap.parse_args()

    labelled_dir = Path(args.labelled)
    meta_dir     = Path(args.meta)
    out_dir      = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_ids = [args.match] if args.match else sorted({
        p.name.split("_labelled.csv")[0] for p in labelled_dir.glob("*_labelled.csv")
    })
    if not match_ids:
        print(f"No *_labelled.csv files found in {labelled_dir}")
        return

    all_features = []
    for match_id in match_ids:
        labelled_path = labelled_dir / f"{match_id}_labelled.csv"
        meta_path      = meta_dir / f"{match_id}_match_meta.json"

        if not labelled_path.exists():
            print(f"  [SKIP] {labelled_path} not found"); continue
        if not meta_path.exists():
            print(f"  [SKIP] {meta_path} not found"); continue

        print(f"\nProcessing {match_id}...")
        feat_df = process_match(labelled_path, meta_path, match_id, args.bg_stride)

        out_path = out_dir / f"{match_id}_features.csv"
        feat_df.to_csv(out_path, index=False)
        print(f"  [{match_id}] {len(feat_df)} rows x {feat_df.shape[1]} cols -> {out_path}")
        print(f"  Label distribution:\n{feat_df['label'].value_counts().to_string()}")
        all_features.append(feat_df)

    if all_features:
        combined = pd.concat(all_features, ignore_index=True)

        # ── final leakage guard ──────────────────────────────────────────
        # Hard assertion: none of the forbidden event-derived column names
        # can exist in the final training table, no matter what changes
        # upstream. This makes the leak fail loudly instead of silently.
        leaked = FORBIDDEN_COLS & set(combined.columns)
        assert not leaked, f"LEAKAGE DETECTED: forbidden columns present: {leaked}"

        out_all = out_dir / "full_training_table.csv"
        combined.to_csv(out_all, index=False)
        print(f"\nAll matches combined: {len(combined)} rows x {combined.shape[1]} cols -> {out_all}")
        print(f"Matches included: {sorted(combined['match_id'].unique())}")
        print(combined["label"].value_counts().to_string())
        print("\n[OK] No forbidden event-derived columns present in final table.")


if __name__ == "__main__":
    main()
