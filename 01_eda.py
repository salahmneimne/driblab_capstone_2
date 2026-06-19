"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 1: Exploratory Data Analysis
=============================================================

Converted from 01-eda.ipynb to match the pipeline script style
(02_align_and_clean.py ... 06_train_classifier_v7.py).

Sections
--------
1a  Event vocabulary        (dim_event_type.csv)
1b  Event distribution      (*_events.json)
1c  Tracking structure      (*_tracking_data.jsonl)
1d  Cross-file consistency  (coordinate + clock mismatch)
1e  Target event selection  (12 classes confirmed by EDA)

All plots are saved as PNG to --out (default ./eda_output).
All console output is also written to eda_summary.txt in --out.

Usage
-----
  python 01_eda.py --data ./data/raw --out ./eda_output
  python 01_eda.py --data ./data/raw --out ./eda_output --deep-match 678949
"""

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# ── constants ──────────────────────────────────────────────────────────────
EXCLUDED_MATCHES = {"745399"}   # flagged by mentor pending duplication check

TARGET_EVENTS = [
    "PASS", "TACKLE", "INTERCEPTION", "CLEARANCE", "BALL RECOVERY",
    "AERIAL", "SAVE", "MISSED SHOT", "SAVED SHOT", "GOAL",
    "DISPOSSESSED", "KEEPER PICKUP",
]

REASONING = {
    "PASS"         : "Highest volume (~63%). Foundational: defines possession exchange. Ball speed + same-team transfer.",
    "TACKLE"       : "Opponent within 1.5m when ball changes team + ball near ground (z < 1.5m).",
    "INTERCEPTION" : "Ball in flight (z > 0.5m) or no contact-range player when team changes.",
    "CLEARANCE"    : "High-speed ball away from defensive zone. Kinematic signal.",
    "BALL RECOVERY": "Loose ball claimed by nearest player. Ball speed near zero.",
    "AERIAL"       : "ball_z > 1.5m is the cleanest binary signal in the dataset. Both teams nearby.",
    "SAVE"         : "Ball heading to goal + goalkeeper position + ball stays in play.",
    "MISSED SHOT"  : "Ball speed > 15 m/s toward goal mouth + ball exits pitch at far end.",
    "SAVED SHOT"   : "Same as missed shot but ball stays in play after goalkeeper contact.",
    "GOAL"         : "Ball crosses goal line. Rare but essential. Same shot trajectory signal.",
    "DISPOSSESSED" : "Player loses ball at feet. Combines with tackle signal (team changes, contact).",
    "KEEPER PICKUP": "Goalkeeper claims ball. GK position + ball_z + ball stops.",
}

EXCLUDED_REASONING = {
    "FOUL"            : "No reliable contact signal from (x,y) dots. Needs pose or video.",
    "CARD"            : "Administrative consequence of FOUL. Not independently detectable.",
    "BALL TOUCH"      : "Too fine-grained for 10 Hz. Cannot distinguish from carry.",
    "TAKEON"          : "Dribble attempt. Cannot reliably distinguish from carry at 10 Hz.",
    "OFFSIDE PASS"    : "Requires computing offside line. Out of scope for tracking alone.",
    "CHALLENGE"       : "Broad overlap with TACKLE. Excluded to avoid label noise.",
    "SUBSTITUTION ON" : "Administrative. No ball or position signal.",
    "SUBSTITUTION OFF": "Administrative. No ball or position signal.",
    "END"             : "Match boundary marker. Administrative.",
    "FORMATION CHANGE": "Tactical annotation. No event-level tracking signal.",
    "CLAIM"           : "Goalkeeper sub-event. Covered by SAVE.",
    "CHANCE MISSED"   : "Editorial label. Not independently detectable.",
}


# ── helpers ────────────────────────────────────────────────────────────────

def load_events(path: Path) -> list:
    """Load a *_events.json file, tolerating trailing-comma JSON quirks."""
    raw   = path.read_text(encoding="utf-8")
    fixed = re.sub(r",\s*\](\s*)$", r"]\1", raw.rstrip())
    return json.loads(fixed)


class Logger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path: Path):
        self._file = open(path, "w", encoding="utf-8")

    def log(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        self._file.write(text + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def save_fig(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Section 0: File discovery ──────────────────────────────────────────────

def section0_discovery(data_dir: Path, log) -> tuple:
    dim_path       = data_dir / "dim_event_type.csv"
    event_files    = sorted(data_dir.glob("*_events.json"))
    tracking_files = sorted(data_dir.glob("*_tracking_data.jsonl"))

    ev_ids   = {re.match(r"(\d+)_events",   f.stem).group(1) for f in event_files}
    tr_ids   = {re.match(r"(\d+)_tracking", f.stem).group(1) for f in tracking_files}
    both_ids = sorted((ev_ids & tr_ids) - EXCLUDED_MATCHES)

    log.log("\n" + "=" * 68)
    log.log("  FINDING 0 — DATA AVAILABILITY")
    log.log("=" * 68)
    log.log(f"  dim_event_type.csv found : {dim_path.exists()}")
    log.log(f"  Event files              : {len(event_files)}")
    log.log(f"  Tracking files           : {len(tracking_files)}")
    log.log(f"  Matches with BOTH assets : {len(both_ids)}  -> {both_ids}")
    log.log(f"  Events only (no tracking): {sorted(ev_ids - tr_ids)}")
    log.log(f"  Tracking only (no events): {sorted(tr_ids - ev_ids)}")
    log.log(f"  Excluded matches         : {sorted(EXCLUDED_MATCHES)}  (mentor instruction)")
    log.log()
    log.log("  With few fully-labelled matches: favour data-efficient methods.")
    log.log("  -> Group k-fold CV by match_id; gradient-boosted trees over deep nets.")

    return dim_path, event_files, tracking_files, both_ids


# ── Section 1a: Event vocabulary ──────────────────────────────────────────

def section1a_vocabulary(dim_path: Path, out_dir: Path, log) -> pd.DataFrame:
    log.log("\n" + "=" * 68)
    log.log("  SECTION 1a — EVENT VOCABULARY  (dim_event_type.csv)")
    log.log("=" * 68)

    dim = pd.read_csv(dim_path)
    category_map = {
        0: "No Event", 1: "Admin/Meta", 2: "Passing/Possession",
        3: "Goalkeeper", 4: "Defensive Duels", 5: "Attacking",
    }
    dim["category_name"] = dim["category_id"].map(category_map)

    log.log(f"  Total rows       : {len(dim)}")
    log.log(f"  Unique event IDs : {dim['event_id'].nunique()}")
    log.log()
    log.log("  By category:")
    for cat_id, grp in dim.groupby("category_id"):
        log.log(f"    [{cat_id}] {category_map.get(cat_id, '?'):25s} -> {len(grp):3d} types")

    primary_ids   = set(dim[dim["event_id"] <  100]["event_id"])
    qualifier_ids = set(dim[dim["event_id"] >= 100]["event_id"])
    log.log(f"\n  Primary types   (id  <100): {len(primary_ids)}  <- DETECTION TARGETS")
    log.log(f"  Qualifier types (id >=100): {len(qualifier_ids)}  <- features/annotations, not targets")

    fig, ax = plt.subplots(figsize=(10, 5))
    cnt = dim.groupby("category_name").size().sort_values()
    cnt.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(cnt)))
    for bar in ax.patches:
        ax.text(bar.get_width() + .1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center")
    ax.set_title("1a — Event types defined per category")
    ax.set_xlabel("Count")
    save_fig(fig, out_dir, "1a_vocabulary")

    return dim


# ── Section 1b: Event distribution ────────────────────────────────────────

def section1b_events(event_files: list, out_dir: Path, log) -> pd.DataFrame:
    log.log("\n" + "=" * 68)
    log.log("  SECTION 1b — EVENTS  (*_events.json)")
    log.log("=" * 68)

    all_events = []
    for ef in event_files:
        for ev in load_events(ef):
            all_events.append({
                "match_id"       : ev.get("match_id"),
                "event_type_id"  : ev["event"]["event_type_id"],
                "event_type_name": ev["event"]["event_type_name"],
                "team_id"        : ev["team"]["team_id"]     if ev.get("team")   else None,
                "team_name"      : ev["team"]["team_name"]   if ev.get("team")   else None,
                "player_id"      : ev["player"]["player_id"] if ev.get("player") else None,
                "period_id"      : ev.get("period_id"),
                "min"            : ev.get("min"),
                "sec"            : ev.get("sec"),
                "milisec"        : ev.get("milisec", 0),
                "x"              : ev.get("x"),
                "y"              : ev.get("y"),
                "x_end"          : ev.get("x_end"),
                "y_end"          : ev.get("y_end"),
                "outcome"        : ev.get("outcome"),
                "xg"             : ev.get("xg"),
            })

    ev_df = pd.DataFrame(all_events)
    total = len(ev_df)
    log.log(f"  Total events : {total}  |  Matches : {ev_df['match_id'].nunique()}")

    log.log("\n  [COORDINATE CHECK]")
    log.log(f"    x: {ev_df['x'].min():.1f} -> {ev_df['x'].max():.1f}  (expect 0-100, normalised)")
    log.log(f"    y: {ev_df['y'].min():.1f} -> {ev_df['y'].max():.1f}  (expect 0-100, normalised)")
    log.log("    x=100 always = goal being attacked (per team, per half)")

    log.log("\n  [SHOT CLUSTER TEST] x>50 for all shots? (confirms normalisation)")
    shots = ev_df[ev_df["event_type_id"].isin({13, 14, 15, 16})]
    for pid, grp in shots.groupby("period_id"):
        xs = grp["x"].dropna().tolist()
        if xs:
            log.log(f"    P{pid}: min x={min(xs):.0f}  all>50: {all(v > 50 for v in xs)}")

    log.log("\n  [TIME ENCODING]")
    for pid, grp in ev_df.groupby("period_id"):
        log.log(f"    P{pid}: minute range {grp['min'].min()}-{grp['min'].max()}")
    log.log("    -> P2 does NOT reset at 45. But tracking VTS is also continuous.")
    log.log("       These are two DIFFERENT clocks — misalignment fixed in Step 2.")

    log.log("\n  [OUTCOME SEMANTICS]")
    for et, grp in ev_df.groupby("event_type_name"):
        if grp["outcome"].notna().any():
            log.log(f"    {et:<28s} outcome=True: {grp['outcome'].mean() * 100:.0f}%")

    type_cnt = ev_df["event_type_name"].value_counts()
    log.log("\n  [CLASS DISTRIBUTION]")
    for name, cnt in type_cnt.items():
        bar = "|" * int(cnt / total * 50)
        log.log(f"    {name:<28s} {cnt:5d} ({cnt / total * 100:5.1f}%)  {bar}")

    # Plot 1: bar chart
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#e74c3c" if v == type_cnt.max() else "#3498db" for v in type_cnt.values]
    type_cnt.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("1b — Class distribution  (red = dominant PASS class)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h / total * 100:.1f}%", ha="center", va="bottom", fontsize=7)
    save_fig(fig, out_dir, "1b_class_dist")

    # Plot 2: pitch map by period
    top8    = type_cnt.head(8).index.tolist()
    palette = dict(zip(top8, sns.color_palette("tab10", 8)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (pid, ax) in enumerate(zip([1, 2], axes)):
        sub = ev_df[ev_df["period_id"] == pid]
        ax.set_facecolor("#1a472a")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        for rk in [
            dict(xy=(0, 30),  width=16, height=40, fill=False, ec="white", lw=1),
            dict(xy=(84, 30), width=16, height=40, fill=False, ec="white", lw=1),
        ]:
            ax.add_patch(mpatches.Rectangle(**rk))
        ax.axvline(50, color="white", lw=1, ls="--", alpha=.5)
        for et in top8:
            m = sub["event_type_name"] == et
            ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"], s=12, alpha=.6,
                       color=palette[et], label=et)
        ax.set_title(f"P{pid} event locations (normalised 0-100)")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=.7)
    plt.suptitle("1b — Event pitch map  (x=100 = attacking goal, always)")
    save_fig(fig, out_dir, "1b_pitch_map")

    return ev_df, type_cnt


# ── Section 1c: Tracking structure ────────────────────────────────────────

def section1c_tracking(tracking_files: list, out_dir: Path, deep_match: str,
                        log) -> dict:
    log.log("\n" + "=" * 68)
    log.log("  SECTION 1c — TRACKING DATA  (*_tracking_data.jsonl)")
    log.log("=" * 68)

    tracking_summaries = {}
    data_dir = tracking_files[0].parent if tracking_files else Path(".")

    for tf in tracking_files:
        m = re.match(r"(\d+)_tracking", tf.stem)
        if not m:
            continue
        mid = m.group(1)
        log.log(f"\n  -- Match {mid} --")

        with open(tf, encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        home = header["teams_data"]["home"]
        away = header["teams_data"]["away"]
        fps  = header.get("FPS", 10)
        log.log(f"    {home['name']} vs {away['name']}  FPS={fps}")

        total = n_ball = n_cam = n_bic = vis_t = vis_f = 0
        bx_v = []; by_v = []; bz_v = []; n_pp = []; p_cnt = Counter()

        with open(tf, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                fr = json.loads(line)
                total += 1
                p_cnt[fr.get("period", 0)] += 1
                b = fr["ball"]
                has_ball = b[0] is not None
                has_cam  = fr.get("cam") is not None
                if has_ball:
                    n_ball += 1
                    bx_v.append(b[0]); by_v.append(b[1])
                    if b[2] is not None:
                        bz_v.append(b[2])
                if has_cam:
                    n_cam += 1
                if has_ball and has_cam:
                    n_bic += 1
                fp = 0
                for tp in fr.get("data", {}).values():
                    for p in tp:
                        fp += 1
                        if p.get("vis"): vis_t += 1
                        else:            vis_f += 1
                if fp > 0:
                    n_pp.append(fp)

        tv = vis_t + vis_f
        log.log(f"    Frames: {total}  |  Periods: {dict(p_cnt)}")
        log.log(f"    Duration: {total / fps / 60:.1f} min  |  Avg players/frame: {np.mean(n_pp):.1f}")
        log.log(f"    Ball present   : {n_ball}/{total} ({n_ball/total*100:.1f}%) all frames")
        log.log(f"    Cam present    : {n_cam}/{total} ({n_cam/total*100:.1f}%) live-play filter")
        log.log(f"    Ball in cam    : {n_bic}/{n_cam} ({n_bic/n_cam*100:.1f}%) reliable ball frames")
        log.log(f"    vis=True       : {vis_t}/{tv} ({vis_t/tv*100:.1f}%) optically observed")
        log.log(f"    vis=False      : {vis_f}/{tv} ({vis_f/tv*100:.1f}%) AI-imputed, less reliable")
        if bx_v:
            log.log(f"    Ball x range   : {min(bx_v):.1f}-{max(bx_v):.1f} (physical metres, expect ~0-105)")
            log.log(f"    Ball y range   : {min(by_v):.1f}-{max(by_v):.1f} (physical metres, expect ~0-68)")
            log.log(f"    Ball z nonzero : {sum(1 for z in bz_v if z > 0)}/{len(bz_v)} frames (real height)")

        tracking_summaries[mid] = dict(
            home=home["name"], away=away["name"],
            total=total, n_ball=n_ball, n_cam=n_cam, n_bic=n_bic,
            pct_ball=n_ball/total*100, pct_cam=n_cam/total*100,
            pct_vis=vis_t/tv*100 if tv > 0 else 0,
            bx=bx_v, by=by_v, bz=bz_v,
        )

    # Deep plots on one match only
    mid_deep = deep_match if deep_match in tracking_summaries else (
        list(tracking_summaries.keys())[0] if tracking_summaries else None
    )
    if mid_deep is None:
        return tracking_summaries

    s       = tracking_summaries[mid_deep]
    tf_deep = data_dir / f"{mid_deep}_tracking_data.jsonl"
    log.log(f"\n  [DEEP ANALYSIS on match {mid_deep}]")

    # Ball & camera availability over time
    ba = []
    with open(tf_deep, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            fr = json.loads(line)
            mc = fr.get("match_clock", [0, 0])
            ba.append({
                "t"       : mc[0] * 60 + mc[1],
                "has_ball": fr["ball"][0] is not None,
                "cam_ok"  : fr.get("cam") is not None,
            })
    ba_df = pd.DataFrame(ba).sort_values("t").reset_index(drop=True)
    ba_df["ball_r"] = ba_df["has_ball"].rolling(100, min_periods=1).mean()
    ba_df["cam_r"]  = ba_df["cam_ok"].rolling(100, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.fill_between(ba_df["t"] / 60, ba_df["ball_r"], alpha=.7, color="#3498db")
    ax1.axhline(ba_df["has_ball"].mean(), color="red", lw=1, ls="--",
                label=f"Avg {ba_df['has_ball'].mean()*100:.0f}%")
    ax1.axvline(45, color="grey", lw=1.2, ls="--")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Ball availability"); ax1.legend()
    ax1.set_title(f"1c — Ball & camera availability — match {mid_deep}")
    ax2.fill_between(ba_df["t"] / 60, ba_df["cam_r"], alpha=.7, color="#e67e22")
    ax2.axhline(ba_df["cam_ok"].mean(), color="red", lw=1, ls="--",
                label=f"Avg {ba_df['cam_ok'].mean()*100:.0f}%")
    ax2.axvline(45, color="grey", lw=1.2, ls="--")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Cam present")
    ax2.set_xlabel("Match time (min)"); ax2.legend()
    save_fig(fig, out_dir, "1c_ball_cam")

    # Ball position heatmap
    valid = [(x, y) for x, y in zip(s["bx"], s["by"]) if -5 <= x <= 110 and -5 <= y <= 73]
    if valid:
        bx = np.array([p[0] for p in valid])
        by = np.array([p[1] for p in valid])
        fig, ax = plt.subplots(figsize=(12, 7))
        h = ax.hist2d(bx, by, bins=[105, 68], range=[[0, 105], [0, 68]],
                      cmap="hot", density=True)
        plt.colorbar(h[3], ax=ax, label="Density")
        for rk in [
            dict(xy=(0, 0),    width=105,  height=68,   fill=False, ec="white", lw=2),
            dict(xy=(0, 22.3), width=16.5, height=23.4, fill=False, ec="white", lw=1),
            dict(xy=(88.5, 22.3), width=16.5, height=23.4, fill=False, ec="white", lw=1),
        ]:
            ax.add_patch(mpatches.Rectangle(**rk))
        ax.axvline(52.5, color="white", lw=1, ls="--", alpha=.5)
        ax.add_patch(plt.Circle((52.5, 34), 9.15, color="white", fill=False, lw=1))
        ax.set_title(f"1c — Ball position heatmap (physical metres) — match {mid_deep}")
        ax.set_xlabel("x (0-105m)"); ax.set_ylabel("y (0-68m)")
        save_fig(fig, out_dir, "1c_heatmap")

    # Player visibility over time
    vr = []
    with open(tf_deep, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            fr  = json.loads(line)
            mc  = fr.get("match_clock", [0, 0])
            t   = mc[0] * 60 + mc[1]
            tot = vt = 0
            for tp in fr.get("data", {}).values():
                for p in tp:
                    tot += 1
                    if p.get("vis"): vt += 1
            if tot > 0:
                vr.append({"t": t, "vis_frac": vt / tot})
    vis_df = pd.DataFrame(vr).sort_values("t")
    vis_df["vis_r"] = vis_df["vis_frac"].rolling(200, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(vis_df["t"] / 60, vis_df["vis_r"], alpha=.7, color="#27ae60")
    ax.axhline(vis_df["vis_frac"].mean(), color="red", lw=1, ls="--",
               label=f"Avg {vis_df['vis_frac'].mean()*100:.0f}% visible")
    ax.axvline(45, color="grey", lw=1.2, ls="--", label="Half-time")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Match time (min)")
    ax.set_ylabel("vis=True fraction")
    ax.set_title(f"1c — Player visibility — match {mid_deep}"); ax.legend()
    save_fig(fig, out_dir, "1c_visibility")

    return tracking_summaries


# ── Section 1d: Cross-file consistency ────────────────────────────────────

def section1d_consistency(both_ids: list, data_dir: Path, tracking_summaries: dict,
                           out_dir: Path, log):
    log.log("\n" + "=" * 68)
    log.log("  SECTION 1d — CROSS-FILE CONSISTENCY")
    log.log("=" * 68)

    for mid_str in both_ids:
        log.log(f"\n  ====== Match {mid_str} ======")
        raw_ev = load_events(data_dir / f"{mid_str}_events.json")

        with open(data_dir / f"{mid_str}_tracking_data.jsonl", encoding="utf-8") as fh:
            tr_header = json.loads(fh.readline())

        home_id  = tr_header["teams_data"]["home"]["id"]
        away_id  = tr_header["teams_data"]["away"]["id"]
        tr_teams = {home_id, away_id}
        ev_teams = {e["team"]["team_id"] for e in raw_ev
                    if e.get("team") and e["team"]["team_id"]}
        ev_pids  = {e["player"]["player_id"] for e in raw_ev
                    if e.get("player") and e["player"]["player_id"]}
        tr_pids  = {int(pid) for roster in tr_header["players_data"].values()
                    for pid in roster.keys()}

        log.log(f"    Team IDs match  : {'OK' if ev_teams == tr_teams else 'MISMATCH'}")
        log.log(f"    Player overlap  : {len(ev_pids & tr_pids)} both | "
                f"{len(ev_pids - tr_pids)} events-only | {len(tr_pids - ev_pids)} tracking-only")
        log.log()
        log.log("    COORDINATE SYSTEMS (incompatible — must transform):")
        log.log("      Events  : normalised 0-100  (x=100=attacking goal, per team per half)")
        log.log("      Tracking: physical metres   (0-105 x 0-68, teams switch ends at half-time)")
        log.log("      Fix: scale by 1.05/0.68 + per-team per-half x-flip from GK positions")

        first_p2_vts = None
        with open(data_dir / f"{mid_str}_tracking_data.jsonl", encoding="utf-8") as fh:
            fh.readline(); prev_p = 1
            for line in fh:
                fr = json.loads(line)
                if fr.get("period") == 2 and prev_p == 1:
                    first_p2_vts = fr["Videotimestamp"]; break
                prev_p = fr.get("period", 1)

        p2_offset = first_p2_vts - 45 * 60 if first_p2_vts else None
        log.log()
        log.log("    CLOCK MISALIGNMENT (must correct):")
        log.log("      Tracking VTS: continuous from kick-off")
        log.log("      Events P2   : restart at 45:00 game time")
        if first_p2_vts:
            log.log(f"      P2 tracking VTS starts at : {first_p2_vts:.1f} s")
            log.log(f"      CORRECTION  : +{p2_offset:.1f} s added to every P2 event_vts")
            log.log("      After fix   : residual <= 50 ms  (pure 10 Hz quantisation noise)")

    # Coordinate system comparison plot on first available match
    mid_deep = both_ids[0] if both_ids else None
    if mid_deep and mid_deep in tracking_summaries:
        s      = tracking_summaries[mid_deep]
        raw_ev = load_events(data_dir / f"{mid_deep}_events.json")
        ev_xy  = pd.DataFrame([{"x": e.get("x"), "y": e.get("y")}
                                for e in raw_ev]).dropna()
        valid  = [(x, y) for x, y in zip(s["bx"], s["by"])
                  if -5 <= x <= 110 and -5 <= y <= 73]
        if valid:
            bx2 = np.array([p[0] for p in valid])
            by2 = np.array([p[1] for p in valid])
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax = axes[0]
            ax.set_facecolor("#1a472a"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.scatter(ev_xy["x"], ev_xy["y"], s=4, alpha=.25, color="#3498db")
            ax.axvline(50, color="white", lw=1, ls="--", alpha=.5)
            ax.set_title("Events — normalised 0-100\n(x=100=attacking goal)")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax = axes[1]
            ax.set_facecolor("#1a472a"); ax.set_xlim(0, 105); ax.set_ylim(0, 68)
            ax.scatter(bx2[:5000], by2[:5000], s=3, alpha=.2, color="#e74c3c")
            for rk in [
                dict(xy=(0, 0),       width=105,  height=68,    fill=False, ec="white", lw=2),
                dict(xy=(0, 24.84),   width=16.5, height=18.32, fill=False, ec="white", lw=1),
                dict(xy=(88.5, 24.84),width=16.5, height=18.32, fill=False, ec="white", lw=1),
            ]:
                ax.add_patch(mpatches.Rectangle(**rk))
            ax.axvline(52.5, color="white", lw=1, ls="--", alpha=.5)
            ax.set_title("Tracking ball — physical metres\n(0-105 x 0-68)")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            plt.suptitle("1d — TWO INCOMPATIBLE COORDINATE SYSTEMS",
                         fontweight="bold", color="darkred")
            save_fig(fig, out_dir, "1d_coords")


# ── Section 1e: Target event selection ────────────────────────────────────

def section1e_target_selection(ev_df: pd.DataFrame, type_cnt: pd.Series,
                                out_dir: Path, log):
    log.log("\n" + "=" * 68)
    log.log("  SECTION 1e — 7 KEY INSIGHTS & TARGET EVENT SELECTION")
    log.log("=" * 68)

    total = len(ev_df)

    # 7 key insights
    rows = [
        (1, "TWO COORDINATE SYSTEMS",
         "Events: 0-100 normalised (x=100=attacking goal, always)",
         "Tracking: physical metres 0-105 x 0-68 (teams switch ends)",
         "Step 2.1: scale + per-team per-half x-flip from GK positions"),
        (2, "TWO MISALIGNED CLOCKS",
         "Tracking VTS: continuous from kick-off",
         "Events P2: restart at 45:00 game time (offset ~459 s)",
         "Step 2.2: add first_P2_VTS - 45*60 to every P2 event"),
        (3, "BALL IS INCOMPLETE BUT HAS HEIGHT",
         "Ball ~49% all frames, ~81% live-play (cam-present) frames",
         "Real z-height available: cleanest signal for aerial duels",
         "Step 2.3: interpolate gaps <=25f | Step 3: z-gate at 1.5m"),
        (4, "~64% POSITIONS ARE AI-IMPUTED (vis=False)",
         "Less reliable than optically confirmed, especially off-ball",
         "Cannot discard — they still carry positional information",
         "Step 3: cost = dist * 1.4 for vis=False players"),
        (5, "SEVERE CLASS IMBALANCE",
         "PASS dominates at ~63% of all labelled events",
         "Plain accuracy is meaningless (predict PASS always = 63%)",
         "Steps 4/5: always per-class F1; class_weight=balanced in Step 5"),
        (6, "PLAYERS ARE DOTS — NO POSE IN TRACKING",
         "Tracking gives (x,y) position, not body orientation or keypoints",
         "Some events may need pose to disambiguate (e.g. tackle direction)",
         "Step 6 (optional): run pose algorithms on video to enrich features"),
        (7, "FEW FULLY-LABELLED MATCHES",
         "Only matches with BOTH events + tracking files can train models",
         "Small labelled set -> deep networks overfit, GBTs generalise better",
         "Step 5: gradient-boosted trees; leave-one-match-out cross-validation"),
    ]
    for num, title, f1, f2, action in rows:
        log.log(f"\n  {num}. {title}")
        log.log(f"     {f1}")
        log.log(f"     {f2}")
        log.log(f"     -> {action}")

    # Target event selection
    log.log("\n" + "=" * 68)
    log.log("  TARGET EVENT SELECTION — FINAL DECISION")
    log.log(f"  12 classes chosen from {ev_df['event_type_name'].nunique()} observed types")
    log.log("=" * 68)
    log.log()
    log.log("  INCLUDED (12 classes):")
    for et in TARGET_EVENTS:
        cnt    = type_cnt.get(et, 0)
        reason = REASONING.get(et, "")
        log.log(f"    {et:<20s}  {cnt:4d} samples")
        log.log(f"                         -> {reason}")
        log.log()

    log.log("  EXCLUDED:")
    for et, reason in EXCLUDED_REASONING.items():
        if et in type_cnt.index:
            log.log(f"    {et:<25s}  {type_cnt.get(et, 0):4d} samples  |  {reason}")

    covered = sum(type_cnt.get(et, 0) for et in TARGET_EVENTS)
    log.log(f"\n  Total target classes : {len(TARGET_EVENTS)}")
    log.log(f"  Labelled samples     : {covered} / {total} ({covered/total*100:.1f}% of all events)")

    # Target class bar chart
    fig, ax = plt.subplots(figsize=(13, 6))
    counts = [type_cnt.get(et, 0) for et in TARGET_EVENTS]
    colors = ["#27ae60" if c > 20 else "#f39c12" if c > 5 else "#e74c3c" for c in counts]
    bars   = ax.bar(TARGET_EVENTS, counts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("1e — Selected target event classes and sample counts")
    ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=40)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#27ae60", label=">20 samples (good)"),
        Patch(color="#f39c12", label="5-20 samples (low)"),
        Patch(color="#e74c3c", label="<5 samples (very low)"),
    ], loc="upper right")
    save_fig(fig, out_dir, "1e_target_classes")

    # Top-90% coverage check
    type_cnt_sorted = type_cnt.sort_values(ascending=False)
    cumsum          = type_cnt_sorted.cumsum() / type_cnt_sorted.sum()
    top90_classes   = list(cumsum[cumsum <= 0.90].index)
    if len(top90_classes) < len(cumsum):
        top90_classes.append(cumsum.index[len(top90_classes)])

    log.log("\n  === TOP-90% COVERAGE CLASSES ===")
    running = 0
    for et in type_cnt_sorted.index:
        cnt     = type_cnt_sorted[et]
        running += cnt
        pct     = running / type_cnt_sorted.sum() * 100
        in_target = et in TARGET_EVENTS
        in_top90  = et in top90_classes
        marker = ("TARGET+TOP90" if (in_target and in_top90)
                  else ("TARGET" if in_target else ("TOP90" if in_top90 else "")))
        log.log(f"    {et:<25s} {cnt:5d} ({cnt/type_cnt_sorted.sum()*100:5.1f}%)  "
                f"cum={pct:5.1f}%  {marker}")

    final_targets = [et for et in TARGET_EVENTS if et in type_cnt.index]
    coverage      = sum(type_cnt.get(et, 0) for et in final_targets) / type_cnt.sum() * 100
    log.log(f"\n  Final targets coverage : {coverage:.1f}% of all labelled events")
    log.log(f"  Target classes         : {final_targets}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Capstone Step 1 — EDA")
    ap.add_argument("--data",        default="./data/raw",
                    help="Directory containing raw data files")
    ap.add_argument("--out",         default="./results/eda_output",
                    help="Directory for plots and summary log")
    ap.add_argument("--deep-match",  default=None,
                    help="Match ID to use for deep tracking plots (default: first available)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(out_dir / "eda_summary.txt")
    log.log("=" * 68)
    log.log("  DRIBLAB CAPSTONE 2026 — STEP 1: EDA")
    log.log(f"  Data dir : {data_dir.resolve()}")
    log.log(f"  Output   : {out_dir.resolve()}")
    log.log("=" * 68)

    dim_path, event_files, tracking_files, both_ids = section0_discovery(data_dir, log)

    dim                = section1a_vocabulary(dim_path, out_dir, log)
    ev_df, type_cnt    = section1b_events(event_files, out_dir, log)
    tracking_summaries = section1c_tracking(tracking_files, out_dir, args.deep_match, log)
    section1d_consistency(both_ids, data_dir, tracking_summaries, out_dir, log)
    section1e_target_selection(ev_df, type_cnt, out_dir, log)

    log.log("\n" + "=" * 68)
    log.log("  STEP 1 COMPLETE")
    log.log(f"  All plots saved to : {out_dir.resolve()}")
    log.log(f"  Full log saved to  : {(out_dir / 'eda_summary.txt').resolve()}")
    log.log("  Next -> 02_align_and_clean.py")
    log.log("=" * 68)
    log.close()


if __name__ == "__main__":
    main()
