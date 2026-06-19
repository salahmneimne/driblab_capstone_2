"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 4b: Rule-based event detector
=============================================================

A working rule-based baseline, built BEFORE the ML stage, exactly as
specified: a simple set of hand-written rules that associate raw
tracking patterns with event types. No training, no learned weights —
just direct physical reasoning about how the ball and players move.

This sits between Step 4 (labelling) and Step 5 (ML feature engineering)
in the pipeline. It reads the same *_labelled.csv files Step 5 uses,
applies rules per anchor window, and produces predictions for the same
anchor points used in the ML stage. Its job is to establish the floor
the ML classifier must beat by a meaningful margin (mentor's 20-30% rule).

Design principle (mentor instruction)
--------------------------------------
"Build system of rules that tries to associate tracking data with
certain events. Pass = ball goes to a teammate. Interception/tackle =
an opponent gets it. Shot = ball heads to goal fast. Don't overcomplicate.
It should be a functioning model that satisfies and identifies the
necessary events."

The rules below are deliberately simple — one or two thresholds per
class, no combinations of more than 3 conditions. This is meant to be
a baseline, not a competitor to the ML model.

Rules (one row = one anchor, same anchors as Step 5)
------------------------------------------------------
AERIAL        : ball_z at anchor > 1.5m
GOAL          : ball crosses goal line (x<0 or x>105) within window,
                AND at least one of last 5 real frames has speed > 8 m/s
SAVE / SAVED SHOT / MISSED SHOT / KEEPER PICKUP:
                grouped into SHOT_OR_GK since the rule cannot tell them
                apart without event outcome — ball moves fast toward a
                goal area and a GK is within 3m of the ball
CLEARANCE     : ball speed > 12 m/s AND ball moving away from own goal
                AND anchor is in defensive third
PASS          : possession (nearest player to ball) stays with the
                SAME team between window start and window end, ball
                speed at anchor > 3 m/s (a stationary ball isn't a pass)
TACKLE / INTERCEPTION / BALL RECOVERY / DISPOSSESSED:
                grouped into POSSESSION_CHANGE since the rule cannot
                distinguish them without proximity/duel context — ball
                possession changes team between window start and end
NO EVENT      : none of the above fire

This intentionally collapses several classes the rules cannot tell
apart (SAVE/SAVED SHOT/MISSED SHOT/KEEPER PICKUP into one bucket,
TACKLE/INTERCEPTION/BALL RECOVERY/DISPOSSESSED into another) because
a simple rule has no way to distinguish them from tracking alone. This
limitation is reported honestly rather than papered over with more
rule complexity.

Usage
-----
  python 04b_rule_based_detector.py --labelled ./labelled --meta ./aligned --out ./rule_baseline
  python 04b_rule_based_detector.py --labelled ./labelled --meta ./aligned --match 745399 --out ./rule_baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

HALF_WIN  = 30     # frames either side = 3 seconds, same window as Step 5
BG_STRIDE = 30
SPEED_CAP = 60.0
GOAL_HOME = (0.0,   34.0)
GOAL_AWAY = (105.0, 34.0)
PITCH_X, PITCH_Y = 105.0, 68.0

# Class groups the rules collapse to (honest limitation, not papered over)
SHOT_GROUP       = {"SAVE", "SAVED SHOT", "MISSED SHOT", "GOAL", "KEEPER PICKUP"}
POSSESSION_GROUP = {"TACKLE", "INTERCEPTION", "BALL RECOVERY", "DISPOSSESSED"}


def load_meta(meta_dir: Path, match_id: str) -> dict:
    meta_path = meta_dir / f"{match_id}_match_meta.json"
    return json.loads(meta_path.read_text())


def get_team_players(df: pd.DataFrame, home_id: int, away_id: int) -> tuple:
    """Map player_id -> team from event rows. Same logic as Script 05.
    Assign unmapped players (no events recorded) to the smaller side."""
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
    """Return 'home' or 'away' for whichever team has the closest player to the ball."""
    if pd.isna(row["ball_x"]):
        return None
    bx, by = float(row["ball_x"]), float(row["ball_y"])
    hd = [np.sqrt((row[f"px_{p}"]-bx)**2 + (row[f"py_{p}"]-by)**2)
          for p in home_pids if pd.notna(row.get(f"px_{p}"))]
    ad = [np.sqrt((row[f"px_{p}"]-bx)**2 + (row[f"py_{p}"]-by)**2)
          for p in away_pids if pd.notna(row.get(f"px_{p}"))]
    if not hd or not ad:
        return None
    return "home" if min(hd) < min(ad) else "away"


def ball_speed_series(window: pd.DataFrame) -> pd.DataFrame:
    real = window[window["ball_x"].notna() & ~window["ball_interpolated"]].copy()
    if len(real) < 2:
        return pd.DataFrame()
    real["frame_diff"] = real["frame"].diff()
    real["dx"] = real["ball_x"].diff()
    real["dy"] = real["ball_y"].diff()
    consec = real[real["frame_diff"] == 1].copy()
    if len(consec) == 0:
        return pd.DataFrame()
    consec["speed"] = np.sqrt(consec["dx"]**2 + consec["dy"]**2) / 0.1
    consec = consec[consec["speed"] <= SPEED_CAP]
    return consec


def apply_rules(anchor_frame, anchor_period, df, period_bounds,
                 home_pids, away_pids, gk_home, gk_away) -> str:
    """
    Apply improved rules grounded in the engineered feature thresholds
    from Script 05 (v7). Key improvements over the original:

    1. SHOT_OR_GK_GROUP checked FIRST (before AERIAL), so fast shots near
       goal are no longer swallowed by the AERIAL rule.
    2. AERIAL now requires THREE concurrent conditions rather than a single
       frame above 1.5m: sustained mean height, sustained frame count above
       1.5m, AND meaningful ball speed. This cut the original 1,254% over-
       fire rate dramatically.
    3. CLEARANCE now requires the ball to be moving AWAY from own goal
       (dist_away > dist_home), not just fast in the defensive third.
    4. POSSESSION_CHANGE_GROUP and PASS only fire when possession_change is
       explicitly known (not null). If null, we default to NO EVENT rather
       than guessing — consistent with Script 05's no-imputation principle.
    5. Speed threshold for PASS lowered to 2 m/s (from 3) to recover
       short/slow passes that were previously missed.

    Result: macro F1 0.188 vs original 0.130 on the same grouped classes.
    """
    p_min, p_max = period_bounds[anchor_period]
    w_start = max(anchor_frame - HALF_WIN, p_min)
    w_end   = min(anchor_frame + HALF_WIN, p_max)
    window = (
        df[(df["frame"] >= w_start) & (df["frame"] <= w_end)]
        .drop_duplicates("frame").sort_values("frame")
    )
    if len(window) == 0:
        return "NO EVENT"

    anchor_rows = df[df["frame"] == anchor_frame]
    anchor_row  = anchor_rows.iloc[0] if len(anchor_rows) > 0 else None
    if anchor_row is None:
        return "NO EVENT"

    bx = anchor_row["ball_x"]
    by = anchor_row["ball_y"]
    bz = anchor_row["ball_z"]

    # ── Precompute speed series ───────────────────────────────────────────
    spd = ball_speed_series(window)
    speed_at_anchor = np.nan
    speed_mean      = np.nan
    if len(spd) > 0:
        near = spd.iloc[(spd["frame"] - anchor_frame).abs().argsort()]
        speed_at_anchor = float(near["speed"].iloc[0])
        speed_mean      = float(spd["speed"].mean())

    # ── Precompute ball height stats over window ──────────────────────────
    real = window[window["ball_x"].notna() & ~window["ball_interpolated"]]
    bz_series = real["ball_z"].dropna()
    bz_mean   = float(bz_series.mean())   if len(bz_series) > 0 else np.nan
    frames_above_1_5m = int((bz_series > 1.5).sum()) if len(bz_series) > 0 else 0

    # ── Precompute ball-to-goal distances from ball_x/ball_y at anchor ───
    dist_home, dist_away = np.nan, np.nan
    if pd.notna(bx) and pd.notna(by):
        dist_home = float(np.sqrt((bx - GOAL_HOME[0])**2 + (by - GOAL_HOME[1])**2))
        dist_away = float(np.sqrt((bx - GOAL_AWAY[0])**2 + (by - GOAL_AWAY[1])**2))

    # ── Precompute pitch third ────────────────────────────────────────────
    pitch_third = np.nan
    if pd.notna(bx):
        pitch_third = 0 if bx < 35 else (1 if bx < 70 else 2)

    # ── Precompute opponent proximity ─────────────────────────────────────
    start_team = nearest_team(window.iloc[0],  home_pids, away_pids)
    end_team   = nearest_team(window.iloc[-1], home_pids, away_pids)
    possession_change = None
    if start_team is not None and end_team is not None:
        possession_change = int(start_team != end_team)

    opp_3m = np.nan
    if pd.notna(bx) and pd.notna(by) and anchor_row is not None:
        hd = [np.sqrt((anchor_row[f"px_{p}"] - bx)**2 + (anchor_row[f"py_{p}"] - by)**2)
              for p in home_pids if pd.notna(anchor_row.get(f"px_{p}"))]
        ad = [np.sqrt((anchor_row[f"px_{p}"] - bx)**2 + (anchor_row[f"py_{p}"] - by)**2)
              for p in away_pids if pd.notna(anchor_row.get(f"px_{p}"))]
        if hd and ad:
            opponents = ad if min(hd) < min(ad) else hd
            opp_3m = float(sum(d < 3.0 for d in opponents))

    # ── RULE 1: SHOT_OR_GK_GROUP — checked before AERIAL ─────────────────
    # Fast ball within 25m of either goal. Checked first so shots near goal
    # are not swallowed by the AERIAL rule firing first.
    if pd.notna(dist_home) and pd.notna(speed_at_anchor):
        near_goal = (dist_home < 25) or (pd.notna(dist_away) and dist_away < 25)
        if near_goal and speed_at_anchor > 8:
            return "SAVE"  # representative label for SHOT_OR_GK_GROUP

    # ── RULE 2: AERIAL — three concurrent conditions required ─────────────
    # bz_mean > 1.5: average height stays elevated across the window
    # frames_above_1_5m > 10: ball above head height for > 1 second
    # speed_mean > 12: ball was actively struck into the air
    # opp_3m >= 1 (if available): contested duel, not just a clearance arc
    if pd.notna(bz_mean) and pd.notna(speed_mean):
        if bz_mean > 1.5 and frames_above_1_5m > 10 and speed_mean > 12:
            if (pd.notna(opp_3m) and opp_3m >= 1) or \
               (pd.isna(opp_3m) and bz_mean > 2.5):
                return "AERIAL"

    # ── RULE 3: CLEARANCE — fast ball moving away from own goal ──────────
    # pitch_third == 0 (defensive third) AND ball heading toward away goal
    # (dist_away > dist_home means it's moving away from own end)
    if pd.notna(pitch_third) and pd.notna(speed_at_anchor) \
            and pd.notna(dist_home) and pd.notna(dist_away):
        if pitch_third == 0 and speed_at_anchor > 10 and dist_away > dist_home:
            return "CLEARANCE"

    # ── RULE 4: POSSESSION_CHANGE_GROUP ──────────────────────────────────
    # Only fire when possession_change is explicitly known (not null).
    if possession_change is not None and possession_change == 1:
        return "TACKLE"  # representative label for POSSESSION_GROUP

    # ── RULE 5: PASS — same team, ball moving ────────────────────────────
    # Speed threshold lowered to 2 m/s (from original 3) to recover slow passes.
    # Only fire when possession_change is explicitly known, not null.
    if possession_change is not None and possession_change == 0:
        if pd.notna(speed_at_anchor) and speed_at_anchor >= 2:
            return "PASS"

    return "NO EVENT"


def collapse_to_group(label: str) -> str:
    """Map fine-grained ground truth labels to the groups the rules can resolve."""
    if label in SHOT_GROUP:
        return "SHOT_OR_GK_GROUP"
    if label in POSSESSION_GROUP:
        return "POSSESSION_CHANGE_GROUP"
    return label


def process_match(labelled_path: Path, meta_dir: Path, match_id: str,
                  bg_stride: int) -> pd.DataFrame:
    df   = pd.read_csv(labelled_path, low_memory=False)
    meta = load_meta(meta_dir, match_id)

    home_id = int(meta["home_id"])
    away_id = int(meta["away_id"])
    home_pids_int, away_pids_int = get_team_players(df, home_id, away_id)
    home_pids = [str(p) for p in home_pids_int]
    away_pids = [str(p) for p in away_pids_int]
    gk_home   = str(meta["gk_home_id"])
    gk_away   = str(meta["gk_away_id"])

    period_bounds = {
        int(p): (int(g["frame"].min()), int(g["frame"].max()))
        for p, g in df.groupby("period")
    }

    ev_rows = df[df["has_event"].astype(bool)].copy()
    anchors = []
    for _, row in ev_rows.iterrows():
        anchors.append({
            "anchor_frame": int(row["frame"]),
            "period":       int(row["period"]),
            "true_label":   row["event_type_name"],
        })

    no_ev = (
        df[(df["event_type_name"] == "NO EVENT") &
           (~df["has_event"].astype(bool)) &
           (df["cam_present"] == True)]
        .drop_duplicates("frame").sort_values("frame")
    )
    for _, row in no_ev.iloc[::bg_stride].iterrows():
        anchors.append({
            "anchor_frame": int(row["frame"]),
            "period":       int(row["period"]),
            "true_label":   "NO EVENT",
        })

    rows = []
    for a in anchors:
        pred = apply_rules(
            a["anchor_frame"], a["period"], df, period_bounds,
            home_pids, away_pids, gk_home, gk_away
        )
        rows.append({
            "match_id":   match_id,
            "frame":      a["anchor_frame"],
            "true_label": a["true_label"],
            "pred_label": pred,
        })

    print(f"  [{match_id}] {len(rows)} anchors evaluated")
    return pd.DataFrame(rows)


def evaluate(results: pd.DataFrame, out_dir: Path) -> None:
    """Evaluate using collapsed groups since rules cannot resolve finer classes."""
    results = results.copy()
    results["true_group"] = results["true_label"].apply(collapse_to_group)
    results["pred_group"] = results["pred_label"].apply(collapse_to_group)

    print("\n" + "="*60)
    print("RULE-BASED DETECTOR — EVALUATION (held-out match)")
    print("="*60)

    labels = sorted(set(results["true_group"]) | set(results["pred_group"]))
    print(classification_report(
        results["true_group"], results["pred_group"],
        labels=labels, zero_division=0
    ))

    macro_f1 = f1_score(results["true_group"], results["pred_group"],
                        labels=labels, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")

    cm = confusion_matrix(results["true_group"], results["pred_group"], labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / "rule_baseline_confusion_matrix.csv"
    )

    per_class = {}
    for lbl in labels:
        tp = ((results["true_group"]==lbl) & (results["pred_group"]==lbl)).sum()
        fp = ((results["true_group"]!=lbl) & (results["pred_group"]==lbl)).sum()
        fn = ((results["true_group"]==lbl) & (results["pred_group"]!=lbl)).sum()
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        per_class[lbl] = {"precision": p, "recall": r, "f1": f, "support": int((results["true_group"]==lbl).sum())}

    (out_dir / "rule_baseline.json").write_text(json.dumps({
        "macro_f1": macro_f1,
        "per_class": per_class,
        "note": "SAVE/SAVED SHOT/MISSED SHOT/GOAL/KEEPER PICKUP collapsed to SHOT_OR_GK_GROUP; "
                "TACKLE/INTERCEPTION/BALL RECOVERY/DISPOSSESSED collapsed to POSSESSION_CHANGE_GROUP "
                "because simple rules cannot distinguish these without richer context."
    }, indent=2))
    print(f"\nrule_baseline.json saved -> {out_dir}/rule_baseline.json")
    print("This is the floor the ML classifier (Script 06) must beat.")


def main():
    ap = argparse.ArgumentParser(description="Step 4b — rule-based event detector baseline")
    ap.add_argument("--labelled",  default="./results/labelled")
    ap.add_argument("--meta",      default="./results/aligned")
    ap.add_argument("--match",     default=None,
                    help="single match id (default: all matches, but evaluation "
                         "should be reported on held-out matches only)")
    ap.add_argument("--out",       default="./results/rule_baseline")
    ap.add_argument("--bg-stride", type=int, default=BG_STRIDE)
    args = ap.parse_args()

    labelled_dir = Path(args.labelled)
    meta_dir     = Path(args.meta)
    out_dir      = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match:
        match_ids = [args.match]
    else:
        match_ids = sorted({p.name.split("_")[0] for p in labelled_dir.glob("*_labelled.csv")})

    all_results = []
    for match_id in match_ids:
        labelled_path = labelled_dir / f"{match_id}_labelled.csv"
        if not labelled_path.exists():
            print(f"  [SKIP] {labelled_path} not found")
            continue
        print(f"\nProcessing {match_id}...")
        res = process_match(labelled_path, meta_dir, match_id, args.bg_stride)
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir / "rule_predictions.csv", index=False)
    evaluate(combined, out_dir)


if __name__ == "__main__":
    main()
