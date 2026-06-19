"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 2: Align + Clean (the foundation)
=============================================================

Turns raw events (normalised 0-100, always-attacking) + raw tracking
(physical 105x68 metres, teams switch ends) into one aligned, cleaned
basis you can build possession + detection on.

What it does, in order:
  1. COORDINATE ALIGNMENT
       - infer each team's attacking direction *per half* from tracking
         (no blind "flip period 2" — both teams attack opposite ways in
          the same half, so direction is per-team-per-period)
       - map event (x,y) and (x_end,y_end) onto the physical pitch with a
         180-degree rotation when the team attacks toward x=0
  2. BALL CLEANING
       - drop physically-impossible ball coordinates (the data has values
         in the thousands), then linearly interpolate only SHORT gaps
  3. CAM / RELIABILITY FLAG
       - mark frames with a camera polygon (the calibrated, reliable ones)
  4. TIME SYNC
       - build a continuous 0.1s game clock per frame and match each event
         to its nearest frame; events that can't be matched within tolerance
         are FLAGGED, not force-matched (this is what exposes the
         second-half offset found in Step 1)
  5. VALIDATION REPORT
       - prints checks so you can SEE whether alignment worked before
         trusting it downstream

Usage
-----
  python 02_align_and_clean.py --data ./data/raw --match 678949 --out ./aligned
  # or import the functions:  from importlib import import_module ...

Field-name assumptions (match your colleague's working loader / real files):
  events frame:  ev["event"]["event_type_id"|"event_type_name"|"id"],
                 ev["team"]["team_id"], ev["player"]["player_id"],
                 ev["period_id"|"min"|"sec"|"milisec"|"x"|"y"|"x_end"|"y_end"|"outcome"]
  tracking hdr:  header["teams_data"]["home"|"away"]{"id","name"},
                 header["players_data"][team_id_str][player_id_str]{"position",...}
  tracking row:  obj["frame"|"period"], obj["match_clock"]=[min,sec],
                 obj["Videotimestamp"], obj["ball"]=[x,y,z],
                 obj["data"]={team_id_str:[{"id","x","y","vis"}]}, obj["cam"]
If a key differs in your files, change it in ONE place (the loaders below).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Pitch constants (physical tracking pitch)
# ----------------------------------------------------------------------------- #
PITCH_X = 105.0
PITCH_Y = 68.0
CENTER_X = PITCH_X / 2.0

# Ball cleaning thresholds
BALL_X_BOUNDS = (-5.0, PITCH_X + 5.0)   # small margin for balls just off the pitch
BALL_Y_BOUNDS = (-5.0, PITCH_Y + 5.0)
MAX_INTERP_GAP_FRAMES = 5               # interpolate ball gaps up to 0.5s (10Hz)

# Time-sync tolerance: a 10Hz frame is 0.1s; event milisec adds sub-second info.
TIME_MATCH_TOL_S = 0.30


# ============================================================================= #
# LOADERS
# ============================================================================= #
def _load_events_json(path: Path) -> list:
    """Load an events JSON that may carry a trailing comma before the ]."""
    raw = path.read_text(encoding="utf-8")
    fixed = re.sub(r",\s*\](\s*)$", r"]\1", raw.rstrip())
    return json.loads(fixed)


def load_events(path: Path) -> pd.DataFrame:
    """Flatten the nested events JSON into a tidy DataFrame."""
    rows = []
    for ev in _load_events_json(path):
        event = ev.get("event", {}) or {}
        team = ev.get("team", {}) or {}
        player = ev.get("player", {}) or {}
        rows.append({
            "event_id":        event.get("id"),
            "event_type_id":   event.get("event_type_id"),
            "event_type_name": event.get("event_type_name"),
            "team_id":         team.get("team_id"),
            "team_name":       team.get("team_name"),
            "player_id":       player.get("player_id"),
            "player_name":     player.get("player_name"),
            "period_id":       ev.get("period_id"),
            "min":             ev.get("min"),
            "sec":             ev.get("sec"),
            "milisec":         ev.get("milisec", 0),
            "x":               ev.get("x"),
            "y":               ev.get("y"),
            "x_end":           ev.get("x_end"),
            "y_end":           ev.get("y_end"),
            "outcome":         ev.get("outcome"),
        })
    df = pd.DataFrame(rows)
    # normalise types
    for col in ("team_id", "player_id", "period_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # absolute event time in seconds (match clock is continuous across halves)
    df["event_abs_sec"] = (
        df["min"].fillna(0) * 60 + df["sec"].fillna(0) + df["milisec"].fillna(0) / 1000.0
    )
    return df


def load_tracking(path: Path):
    """
    Returns
    -------
    header        : dict (first line)
    frames        : DataFrame, one row per frame (ball/cam/clock, NOT players)
    players       : DataFrame, one row per player per frame
                    columns: frame, period, team_id, player_id, px, py, vis
    dir_stats     : dict[(period, team_id)][player_id] -> [sum_x, count]
                    (vis=True player x positions, used to infer attacking side)

    NOTE: player positions (px, py) are in the SAME physical coordinate system
    as the ball (metres, 105x68, teams switch ends at half-time). They are NOT
    yet attacking-direction normalised — that normalisation is applied to events
    only in transform_event_coords. Use vis=True rows for reliable positions;
    vis=False rows are AI-imputed and less reliable off the ball.
    """
    with open(path, encoding="utf-8") as fh:
        header = json.loads(fh.readline())

        frame_rows  = []
        player_rows = []
        dir_stats   = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            period = obj.get("period")
            frame  = obj.get("frame")
            mc = obj.get("match_clock") or [None, None]
            clock_sec = (mc[0] * 60 + mc[1]) if (mc[0] is not None and mc[1] is not None) else np.nan
            vts = obj.get("Videotimestamp")

            ball = obj.get("ball") or [None, None, None]
            bx = ball[0] if len(ball) > 0 else None
            by = ball[1] if len(ball) > 1 else None
            bz = ball[2] if len(ball) > 2 else None

            frame_rows.append({
                "frame":      frame,
                "period":     period,
                "clock_sec":  clock_sec,
                "video_ts":   vts if vts is not None else np.nan,
                "ball_x_raw": bx if bx is not None else np.nan,
                "ball_y_raw": by if by is not None else np.nan,
                "ball_z_raw": bz if bz is not None else np.nan,
                "cam_present": obj.get("cam") is not None,
            })

            # collect every player position for feature engineering
            # AND accumulate vis=True positions for attacking-direction inference
            for team_id_str, players in (obj.get("data") or {}).items():
                try:
                    team_id = int(team_id_str)
                except (TypeError, ValueError):
                    continue
                for p in players:
                    px  = p.get("x")
                    py  = p.get("y")
                    vis = bool(p.get("vis", False))
                    pid = p.get("id")

                    # store every player regardless of vis (Step 5 can filter)
                    if px is not None and py is not None:
                        player_rows.append({
                            "frame":     frame,
                            "period":    period,
                            "team_id":   team_id,
                            "player_id": pid,
                            "px":        float(px),
                            "py":        float(py),
                            "vis":       vis,
                        })

                    # direction inference uses vis=True only (reliable positions)
                    if vis and px is not None:
                        slot = dir_stats[(period, team_id)][pid]
                        slot[0] += float(px)
                        slot[1] += 1

    frames  = pd.DataFrame(frame_rows).sort_values(["period", "frame"]).reset_index(drop=True)
    players = pd.DataFrame(player_rows).sort_values(["frame", "team_id", "player_id"]).reset_index(drop=True)
    return header, frames, players, dir_stats


# ============================================================================= #
# 1. COORDINATE ALIGNMENT
# ============================================================================= #
def infer_attacking_directions(dir_stats, min_player_samples=200):
    """
    For each (period, team_id) decide which way the team attacks, from tracking.

    Logic: a team's deepest player (their GK) sits on their own goal line.
    If that deepest extreme is near x=0 the team defends the left goal and
    therefore ATTACKS RIGHT (+1, attacking goal at x=105). If it's near x=105
    they ATTACK LEFT (-1, attacking goal at x=0).

    Returns
    -------
    directions : dict[(period, team_id)] -> +1 (toward 105) or -1 (toward 0)
    detail     : dict[(period, team_id)] -> dict with the evidence used
    """
    directions, detail = {}, {}
    for (period, team_id), players in dir_stats.items():
        means = [s[0] / s[1] for s in players.values() if s[1] > 0]
        n = sum(s[1] for s in players.values())
        if len(means) < 2 or n < min_player_samples:
            continue
        lo, hi = min(means), max(means)
        dist_lo, dist_hi = CENTER_X - lo, hi - CENTER_X
        direction = +1 if dist_lo >= dist_hi else -1
        directions[(period, team_id)] = direction
        detail[(period, team_id)] = {
            "deepest_x_low": round(lo, 1),
            "deepest_x_high": round(hi, 1),
            "direction": "→105 (right)" if direction == +1 else "→0 (left)",
            "samples": int(n),
        }
    return directions, detail


def _rotate(x_norm, y_norm, direction):
    """Map a normalised (0-100, attacking->100) point to physical metres."""
    if pd.isna(x_norm) or pd.isna(y_norm) or direction is None:
        return np.nan, np.nan
    xm = x_norm / 100.0 * PITCH_X
    ym = y_norm / 100.0 * PITCH_Y
    if direction == -1:                 # attacking toward x=0 -> 180-degree rotation
        xm = PITCH_X - xm
        ym = PITCH_Y - ym
    return xm, ym


def transform_event_coords(events: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Add physical-metre columns x_m, y_m, x_end_m, y_end_m to events."""
    df = events.copy()
    dirs = df.apply(
        lambda r: directions.get((r["period_id"], r["team_id"])), axis=1
    )
    df["attack_dir"] = dirs

    start = [_rotate(x, y, d) for x, y, d in zip(df["x"], df["y"], dirs)]
    end = [_rotate(x, y, d) for x, y, d in zip(df["x_end"], df["y_end"], dirs)]
    df["x_m"], df["y_m"] = zip(*start)
    df["x_end_m"], df["y_end_m"] = zip(*end)
    return df


# ============================================================================= #
# 2. BALL CLEANING
# ============================================================================= #
def clean_ball(frames: pd.DataFrame) -> pd.DataFrame:
    """
    Remove out-of-bounds ball coordinates, then interpolate only short gaps.
    Adds ball_x/ball_y/ball_z (clean) and ball_interpolated (bool).
    """
    df = frames.copy()

    in_x = df["ball_x_raw"].between(*BALL_X_BOUNDS)
    in_y = df["ball_y_raw"].between(*BALL_Y_BOUNDS)
    valid = in_x & in_y & df["ball_x_raw"].notna() & df["ball_y_raw"].notna()

    df["ball_x"] = np.where(valid, df["ball_x_raw"], np.nan)
    df["ball_y"] = np.where(valid, df["ball_y_raw"], np.nan)
    df["ball_z"] = np.where(valid, df["ball_z_raw"], np.nan)
    df["_outlier"] = (~valid) & (df["ball_x_raw"].notna())   # had a value but it was junk

    df["ball_interpolated"] = False
    # interpolate within each period so gaps don't bridge half-time
    parts = []
    for _, g in df.groupby("period", sort=False):
        g = g.copy()
        was_nan = g["ball_x"].isna()
        gap_id = (was_nan != was_nan.shift()).cumsum()
        for _, idx in g.index.to_series().groupby(gap_id):
            block = g.loc[idx]
            if block["ball_x"].isna().all() and 0 < len(block) <= MAX_INTERP_GAP_FRAMES:
                g.loc[idx, "ball_interpolated"] = True
        for col in ("ball_x", "ball_y", "ball_z"):
            g[col] = g[col].interpolate(method="linear", limit=MAX_INTERP_GAP_FRAMES,
                                        limit_area="inside")
        # only keep interpolation flag where we actually filled a short gap
        g.loc[g["ball_x"].notna() & was_nan & ~g["ball_interpolated"], "ball_interpolated"] = False
        parts.append(g)
    df = pd.concat(parts).sort_index()
    return df


# ============================================================================= #
# 4. TIME SYNC
# ============================================================================= #
def build_game_clock(frames: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous 0.1s game clock per frame.

    Videotimestamp is high-resolution but restarts each half (separate video
    files), so per period we shift it by the median offset to the coarse
    (1s) match clock:  game_clock = video_ts + median(clock_sec - video_ts).
    Falls back to clock_sec where video_ts is missing.
    """
    df = frames.copy()
    df["game_clock"] = np.nan
    for period, g in df.groupby("period", sort=False):
        usable = g["video_ts"].notna() & g["clock_sec"].notna()
        if usable.any():
            offset = float((g.loc[usable, "clock_sec"] - g.loc[usable, "video_ts"]).median())
            gc = g["video_ts"] + offset
            gc = gc.fillna(g["clock_sec"])
        else:
            gc = g["clock_sec"]
        df.loc[g.index, "game_clock"] = gc
    return df


def _nearest_match(shifted_t, gc_sorted):
    """For each time in shifted_t, find the index into gc_sorted of the nearest value."""
    pos = np.searchsorted(gc_sorted, shifted_t)
    pos = np.clip(pos, 1, len(gc_sorted) - 1)
    left, right = gc_sorted[pos - 1], gc_sorted[pos]
    choose_left = (shifted_t - left) <= (right - shifted_t)
    nearest = np.where(choose_left, pos - 1, pos)
    residual = np.abs(shifted_t - gc_sorted[nearest])
    return nearest, residual


# Diagnostic only: median distance (metres) between an event's ball location
# and the tracking ball position at its matched frame. Baseline noise between
# event coordinates and tracking ball positions runs ~25-30m even at the
# correct time offset, so this is just a sanity check, not a calibration signal.
SPATIAL_DIST_DIAGNOSTIC_TOL_M = 40.0


def _calibrate_period_offset(period, gc_sorted):
    """
    Constant additive offset (seconds) to map event_abs_sec -> game_clock for
    one period.

    events.json uses real football-clock convention: period p starts at
    (p-1)*45 minutes. tracking's game_clock runs continuously across periods
    (no halftime reset), so game_clock at the first frame of period p
    corresponds to event time (p-1)*45*60. The offset is the difference:

        offset_p = game_clock[first frame of period p] - (p-1)*45*60

    This is deterministic and per-match (no search needed). Verified against
    a brute-force spatial search (event ball location vs tracking ball
    position): for period 1 the offset is ~0.5s (within the noise floor of
    the spatial check), and for period 2 it matches the spatial search to
    within 0.06s.
    """
    return float(gc_sorted[0]) - (period - 1) * 45 * 60


def _spatial_diagnostic(ev_x, ev_y, ball_x_sorted, ball_y_sorted, nearest, time_residual, tol_s):
    """Median distance (m) between event ball location and matched-frame ball position."""
    bx, by = ball_x_sorted[nearest], ball_y_sorted[nearest]
    valid = ~np.isnan(ev_x) & ~np.isnan(ev_y) & ~np.isnan(bx) & ~np.isnan(by) & (time_residual <= tol_s)
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    dist = np.hypot(bx[valid] - ev_x[valid], by[valid] - ev_y[valid])
    return float(np.median(dist)), n


def sync_event_time(events: pd.DataFrame, frames: pd.DataFrame,
                    tol_s=TIME_MATCH_TOL_S) -> pd.DataFrame:
    """
    Match each event to its nearest tracking frame within the same period.

    Events' clock (min/sec) follows real football-clock convention, which
    restarts near minute 45 at the start of period 2, while tracking's
    game_clock runs continuously across periods. We first calibrate a
    constant per-period offset (see _calibrate_period_offset) and apply it
    to event_abs_sec before matching. Events whose nearest frame is still
    farther than tol_s after calibration are flagged (matched_frame stays
    set, but time_matched=False).
    """
    df = events.copy()
    df["matched_frame"] = pd.NA
    df["matched_frame_idx"] = pd.NA
    df["time_residual_s"] = np.nan
    df["time_matched"] = False
    df["period_clock_offset_s"] = np.nan
    df["period_offset_spatial_dist_m"] = np.nan
    df["period_offset_n_anchors"] = np.nan

    for period, g_fr in frames.groupby("period", sort=False):
        gc = g_fr["game_clock"].to_numpy()
        order = np.argsort(gc)
        gc_sorted = gc[order]
        frame_ids = g_fr["frame"].to_numpy()[order]
        frame_idx = g_fr.index.to_numpy()[order]
        ball_x_sorted = g_fr["ball_x"].to_numpy()[order]
        ball_y_sorted = g_fr["ball_y"].to_numpy()[order]

        ev_mask = df["period_id"] == period
        ev = df.loc[ev_mask]
        if ev.empty or len(gc_sorted) == 0:
            continue

        ev_t = ev["event_abs_sec"].to_numpy()
        ev_x = ev["x_m"].to_numpy()
        ev_y = ev["y_m"].to_numpy()
        offset = _calibrate_period_offset(period, gc_sorted)
        nearest, residual = _nearest_match(ev_t + offset, gc_sorted)
        med_dist, n_anchors = _spatial_diagnostic(
            ev_x, ev_y, ball_x_sorted, ball_y_sorted, nearest, residual, tol_s)

        df.loc[ev_mask, "matched_frame"] = frame_ids[nearest]
        df.loc[ev_mask, "matched_frame_idx"] = frame_idx[nearest]
        df.loc[ev_mask, "time_residual_s"] = residual
        df.loc[ev_mask, "time_matched"] = residual <= tol_s
        df.loc[ev_mask, "period_clock_offset_s"] = offset
        df.loc[ev_mask, "period_offset_spatial_dist_m"] = med_dist
        df.loc[ev_mask, "period_offset_n_anchors"] = n_anchors
    return df


# ============================================================================= #
# 5. VALIDATION REPORT
# ============================================================================= #
def validate(events_aligned, frames_clean, directions, detail,
             shot_type_ids=(13, 14, 15, 16)):
    print("\n" + "=" * 65)
    print("  STEP 2 VALIDATION REPORT")
    print("=" * 65)

    # --- attacking directions + per-half flip check ---
    print("\n[1] Attacking direction per team per half")
    teams = sorted({t for (_, t) in directions})
    for t in teams:
        d1 = directions.get((1, t))
        d2 = directions.get((2, t))
        flip = "OK (flips)" if (d1 is not None and d2 is not None and d1 != d2) \
               else "!! CHECK — does not flip"
        print(f"    team {t}:  P1 {detail.get((1,t),{}).get('direction','?'):>12}"
              f"   P2 {detail.get((2,t),{}).get('direction','?'):>12}   {flip}")

    # --- coordinate sanity: shots should sit near the attacking goal ---
    print("\n[2] Coordinate check — shot distance to attacking goal (metres)")
    shots = events_aligned[events_aligned["event_type_id"].isin(shot_type_ids)].copy()
    shots = shots.dropna(subset=["x_m", "attack_dir"])
    if len(shots):
        goal_x = np.where(shots["attack_dir"] == 1, PITCH_X, 0.0)
        dist = np.hypot(shots["x_m"] - goal_x, shots["y_m"] - PITCH_Y / 2)
        print(f"    shots checked     : {len(shots)}")
        print(f"    median dist-to-goal: {np.median(dist):.1f} m  (small = aligned)")
        print(f"    90th pct dist      : {np.percentile(dist,90):.1f} m")
        if np.median(dist) > 35:
            print("    !! median is large — direction/flip may be wrong; inspect.")
    else:
        print("    (no shot-type events found with these ids — adjust shot_type_ids)")

    # --- time sync, with explicit second-half offset detection ---
    print("\n[3] Time sync — residual between event clock and tracking frame")
    for period, g in events_aligned.groupby("period_id"):
        matched = g["time_matched"].mean() * 100
        med = g["time_residual_s"].median()
        offset = g["period_clock_offset_s"].iloc[0]
        spatial = g["period_offset_spatial_dist_m"].iloc[0]
        n_anchors = g["period_offset_n_anchors"].iloc[0]
        print(f"    period {period}: matched within tol = {matched:5.1f}%"
              f"   median residual = {med:.3f}s   calibrated offset = {offset:+.2f}s"
              f"   (n={len(g)})")
        print(f"               calibration quality: median ball-position distance "
              f"= {spatial:.1f}m   (n_anchors={int(n_anchors)})")
        if spatial > SPATIAL_DIST_DIAGNOSTIC_TOL_M:
            print("    !! calibration anchor distance is large — offset may be unreliable, inspect.")

    # --- ball coverage before/after cleaning ---
    print("\n[4] Ball cleaning")
    raw_valid = frames_clean["ball_x_raw"].notna().mean() * 100
    outliers = int(frames_clean["_outlier"].sum())
    clean_valid = frames_clean["ball_x"].notna().mean() * 100
    interp = int(frames_clean["ball_interpolated"].sum())
    print(f"    frames with a raw ball value : {raw_valid:5.1f}%")
    print(f"    out-of-bounds values removed : {outliers}")
    print(f"    short gaps interpolated      : {interp} frames")
    print(f"    frames with a usable ball    : {clean_valid:5.1f}%  (after cleaning)")

    print("\n[5] Camera reliability")
    print(f"    cam-present frames           : {frames_clean['cam_present'].mean()*100:5.1f}%"
          f"   (your reliable-frame filter)")
    print("=" * 65 + "\n")


# ============================================================================= #
# ORCHESTRATION
# ============================================================================= #
def align_match(data_dir: Path, match_id: str):
    ev_path = data_dir / f"{match_id}_events.json"
    tr_path = data_dir / f"{match_id}_tracking_data.jsonl"

    events = load_events(ev_path)
    header, frames, players, dir_stats = load_tracking(tr_path)

    directions, detail = infer_attacking_directions(dir_stats)
    events = transform_event_coords(events, directions)

    frames = clean_ball(frames)
    frames = build_game_clock(frames)
    events = sync_event_time(events, frames)

    # Pivot player positions wide and merge into frames.
    # Each player gets three columns: px_{player_id}, py_{player_id}, vis_{player_id}
    # This keeps 1 row per frame and makes player data available in training_data.csv.
    if not players.empty:
        pivot = players.pivot_table(
            index="frame",
            columns="player_id",
            values=["px", "py", "vis"],
            aggfunc="first",
        )
        pivot.columns = [f"{v}_{p}" for v, p in pivot.columns]
        pivot = pivot.reset_index()
        frames = frames.merge(pivot, on="frame", how="left")
        # cast vis_ columns to bool (pivot_table can produce mixed types)
        for col in frames.columns:
            if col.startswith("vis_"):
                frames[col] = frames[col].astype("boolean")

    return events, frames, directions, detail


def main():
    ap = argparse.ArgumentParser(description="Capstone Step 2 — align + clean")
    ap.add_argument("--data", default="./data/raw")
    ap.add_argument("--match", help="match id, e.g. 678949 (default: all matches found in --data)")
    ap.add_argument("--out", default="./results/aligned")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.match:
        match_ids = [args.match]
    else:
        match_ids = sorted({
            p.name.split("_events.json")[0] for p in data_dir.glob("*_events.json")
        })

    for match_id in match_ids:
        events, frames, directions, detail = align_match(data_dir, match_id)
        validate(events, frames, directions, detail)

        ev_out = out_dir / f"{match_id}_events_aligned.csv"
        fr_out = out_dir / f"{match_id}_tracking_clean.csv"
        events.to_csv(ev_out, index=False)
        frames.drop(columns=["_outlier"]).to_csv(fr_out, index=False)
        n_player_cols = len([c for c in frames.columns if c.startswith("px_")])
        print(f"  saved: {ev_out}")
        print(f"  saved: {fr_out}  ({n_player_cols} player px_ columns included)")

        # write match metadata: team IDs and GK player IDs from JSONL header
        tr_path = data_dir / f"{match_id}_tracking_data.jsonl"
        with open(tr_path, encoding="utf-8") as _fh:
            _hdr = json.loads(_fh.readline())
        meta = {
            "home_id":    _hdr["teams_data"]["home"]["id"],
            "away_id":    _hdr["teams_data"]["away"]["id"],
            "home_name":  _hdr["teams_data"]["home"]["name"],
            "away_name":  _hdr["teams_data"]["away"]["name"],
            "gk_home_id": None,
            "gk_away_id": None,
        }
        for team_id_str, players_meta in _hdr["players_data"].items():
            for pid_str, pdata in players_meta.items():
                if pdata.get("position") == "GK":
                    if int(team_id_str) == meta["home_id"]:
                        meta["gk_home_id"] = int(pid_str)
                    else:
                        meta["gk_away_id"] = int(pid_str)
        meta_out = out_dir / f"{match_id}_match_meta.json"
        meta_out.write_text(json.dumps(meta, indent=2))
        print(f"  saved: {meta_out}")


if __name__ == "__main__":
    main()
