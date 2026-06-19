# Driblab Capstone 2026 — Football Event Detection from Tracking Data

**IE Business School · Sports Analytics Capstone**

A complete end-to-end machine learning pipeline that detects and classifies football events (passes, tackles, aerials, shots, etc.) from 10Hz x/y tracking data across 33 UEFA Champions League matches. The system uses only the physical movement of players and the ball — no video, no pose estimation, no event annotation at prediction time.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Folder Structure](#folder-structure)
3. [Setup](#setup)
4. [Raw Data — What You Need](#raw-data--what-you-need)
5. [Running the Pipeline](#running-the-pipeline)
   - [Step 1 — EDA](#step-1--eda)
   - [Step 2 — Align and Clean](#step-2--align-and-clean)
   - [Step 3 — Join Tracking and Events](#step-3--join-tracking-and-events)
   - [Step 4 — Label Windows](#step-4--label-windows)
   - [Step 4b — Rule-Based Detector](#step-4b--rule-based-detector)
   - [Step 5 — Feature Engineering](#step-5--feature-engineering)
   - [Step 6 — Train Classifiers](#step-6--train-classifiers)
6. [What Gets Produced](#what-gets-produced)
7. [The 5 Engineered Features](#the-5-engineered-features)
8. [Target Event Classes](#target-event-classes)
9. [Results](#results)
10. [Known Limitations](#known-limitations)

---

## Project Goal

Build a system that reads raw player and ball tracking data (x/y/z positions at 10 frames per second) and predicts which football event, if any, is occurring at each moment. The system learns entirely from tracking movement — it never reads event annotations as model inputs.

---

## Folder Structure

Your `driblab_capstone_2` folder on Desktop must be structured exactly as follows before running the pipeline. Scripts will automatically create the `results/` subfolders as they run — you do not need to create them manually.

```
driblab_capstone_2/               <- your project root, run all commands from here
│
├── 01_eda.py
├── 02_align_and_clean.py
├── 03_join_tracking_events.py
├── 04_label_windows.py
├── 04b_rule_based_detector.py
├── 05_feature_engineering_v7.py
├── 06_train_classifier_v7.py
│
├── dim_event_type.csv            <- provided by Driblab, place here at root level
│
├── data/
│   └── raw/                      <- place ALL raw Driblab files here
│       ├── dim_event_type.csv    <- also needs to be here for Script 01
│       ├── 678949_events.json
│       ├── 678949_tracking_data.jsonl
│       ├── 679088_events.json
│       ├── 679088_tracking_data.jsonl
│       └── ...                   <- one _events.json + one _tracking_data.jsonl per match
│
├── results/                      <- created automatically by the scripts
│   ├── eda_output/               <- Script 01 output
│   ├── aligned/                  <- Script 02 output
│   ├── joined/                   <- Script 03 output
│   ├── labelled/                 <- Script 04 output
│   ├── rule_baseline/            <- Script 04b output
│   ├── features_v7/              <- Script 05 output
│   └── models_v7/                <- Script 06 output
│
├── README.md
└── requirements.txt
```

> **Important:** Raw data files (`.jsonl`, `_events.json`) are excluded from GitHub via `.gitignore` because they exceed GitHub's 100MB file size limit. Share the `data/raw/` folder with teammates via Google Drive or another file-sharing service.

---

## Setup

**Step 1 — Make sure Python 3.9 or higher is installed**

```bash
python3 --version
```

**Step 2 — Navigate to your project folder**

```bash
cd ~/Desktop/driblab_capstone_2
```

Run every command in this README from this folder. Never run from a subfolder.

**Step 3 — Create and activate a virtual environment**

```bash
python3 -m venv driblab_env
source driblab_env/bin/activate
```

You will see `(driblab_env)` appear at the start of your terminal prompt. You must activate the environment every time you open a new terminal session before running any script.

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Raw Data — What You Need

Place the following files inside `data/raw/` before running anything:

| File | Description |
|---|---|
| `dim_event_type.csv` | Event type definitions — also place a copy at the project root |
| `<match_id>_events.json` | One per match — event annotations |
| `<match_id>_tracking_data.jsonl` | One per match — 10Hz tracking frames |

Every match needs both its `_events.json` and `_tracking_data.jsonl` file. Matches missing either file are automatically skipped by the pipeline.

> Match `745399` is currently flagged as potentially duplicated and is pending confirmation from the mentor. It is included in all runs for now.

---

## Running the Pipeline

**Always run from the project root:**

```bash
cd ~/Desktop/driblab_capstone_2
source driblab_env/bin/activate
```

Every script has sensible defaults and can be run with no arguments at all. The argument list is provided below each command for reference if you need to override a path.

---

### Step 1 — EDA

**What it does:** Analyses all raw data files. Checks coordinate systems, clock alignment, class distribution, ball availability, and selects the 12 target event classes. Saves 8 plots and a full summary log.

**Run:**

```bash
python3 01_eda.py
```

**Full argument list (all optional):**

```
--data        Path to raw data folder           default: ./data/raw
--out         Where to save plots and log        default: ./results/eda_output
--deep-match  Match ID for detailed plots        default: first available match
```

**Produces in `results/eda_output/`:**

```
eda_summary.txt
1a_vocabulary.png
1b_class_dist.png
1b_pitch_map.png
1c_ball_cam.png
1c_heatmap.png
1c_visibility.png
1d_coords.png
1e_target_classes.png
```

---

### Step 2 — Align and Clean

**What it does:** Reads raw `_events.json` and `_tracking_data.jsonl` files. Converts event coordinates from normalised 0–100 to physical metres, infers each team's attacking direction per half from goalkeeper positions, cleans ball tracking (drops impossible coordinates, interpolates short gaps), syncs event timestamps to tracking frames, and pivots all player positions wide. Produces one output per match.

**Run:**

```bash
python3 02_align_and_clean.py
```

**Full argument list (all optional):**

```
--data        Path to raw data folder           default: ./data/raw
--out         Where to save aligned outputs      default: ./results/aligned
--match       Single match ID to process         default: all matches in --data
```

**Produces in `results/aligned/` — three files per match:**

```
<match_id>_tracking_clean.csv
<match_id>_events_aligned.csv
<match_id>_match_meta.json
```

> This step must complete successfully before any subsequent step can run. If a match fails here, it will be absent from all downstream outputs.

---

### Step 3 — Join Tracking and Events

**What it does:** Left-joins every tracking frame with any event matched to it. Frames with no event get `has_event=False`. Frames with multiple simultaneous events (e.g. both players in an aerial duel) produce multiple rows, one per event.

**Run:**

```bash
python3 03_join_tracking_events.py
```

**Full argument list (all optional):**

```
--aligned     Path to Script 02 output           default: ./results/aligned
--out         Where to save joined outputs        default: ./results/joined
--match       Single match ID to process          default: all matches in --aligned
```

**Produces in `results/joined/` — one file per match:**

```
<match_id>_training_data.csv
```

---

### Step 4 — Label Windows

**What it does:** Fills `event_type_name` and `event_type_id` for every row. The 12 target classes keep their label. Non-target events (FOUL, SUBSTITUTION, etc.) are collapsed to `NO EVENT`. Frames within ±3 frames (300ms) of a target event inherit that event's label. Everything else becomes `NO EVENT`. Runs sanity checks after each match and will fail loudly if labels are inconsistent.

**Run:**

```bash
python3 04_label_windows.py
```

**Full argument list (all optional):**

```
--joined      Path to Script 03 output           default: ./results/joined
--out         Where to save labelled outputs      default: ./results/labelled
--match       Single match ID to process          default: all matches in --joined
--half-win    Label propagation window in frames  default: 3 (= 300ms)
```

**Produces in `results/labelled/` — one file per match:**

```
<match_id>_labelled.csv
```

---

### Step 4b — Rule-Based Detector

**What it does:** A standalone rule-based classifier that predicts events using hand-written thresholds on tracking data. No training involved. Runs on the same anchor windows as the ML model, evaluates against true labels, and saves results. Its macro F1 score is the floor the ML model is compared against.

This step runs in parallel with Scripts 05 and 06 — it does not feed into Script 05 or 06.

**Rules applied:**
- `SHOT_OR_GK_GROUP` — fast ball (speed > 8 m/s) within 25m of either goal, checked first
- `AERIAL` — ball mean height > 1.5m AND above 1.5m for > 10 frames AND mean speed > 12 m/s
- `CLEARANCE` — speed > 10 m/s in defensive third AND ball moving away from own goal
- `POSSESSION_CHANGE_GROUP` — team with ball flips between window start and end
- `PASS` — possession stays same team AND speed > 2 m/s at anchor
- `NO EVENT` — none of the above fire

**Run:**

```bash
python3 04b_rule_based_detector.py
```

**Full argument list (all optional):**

```
--labelled    Path to Script 04 output           default: ./results/labelled
--meta        Path to Script 02 output           default: ./results/aligned
--out         Where to save rule baseline         default: ./results/rule_baseline
--match       Single match ID to process          default: all matches in --labelled
--bg-stride   Background sampling stride          default: 30
```

**Produces in `results/rule_baseline/`:**

```
rule_predictions.csv
rule_baseline.json
rule_baseline_confusion_matrix.csv
```

---

### Step 5 — Feature Engineering

**What it does:** For every real event anchor and every stride-sampled background point (camera-visible frames only), computes 5 feature groups over a ±30-frame (3-second) window. Every feature is derived from tracking data only — no event annotation is read as a feature input. Includes a hard assertion at runtime that will crash the script if any forbidden event-derived column ever appears in the output.

**The 5 feature groups (12 columns total):**

| Group | Columns |
|---|---|
| Ball speed | `ball_speed_mean`, `ball_speed_max`, `ball_speed_at_anchor` |
| Ball height | `ball_z_max`, `ball_z_mean`, `frames_z_above_1_5m` |
| Ball-to-goal distance | `ball_dist_to_home_goal`, `ball_dist_to_away_goal`, `ball_pitch_third` |
| Possession change | `possession_change_in_window` |
| Opponent proximity | `n_opponents_within_1m5`, `n_opponents_within_3m` |

**Run:**

```bash
python3 05_feature_engineering_v7.py
```

**Full argument list (all optional):**

```
--labelled    Path to Script 04 output           default: ./results/labelled
--meta        Path to Script 02 output           default: ./results/aligned
--out         Where to save feature files         default: ./results/features_v7
--match       Single match ID to process          default: all matches in --labelled
--bg-stride   Background sampling stride          default: 12
```

**Produces in `results/features_v7/`:**

```
<match_id>_features.csv        <- one per match
full_training_table.csv        <- all 33 matches concatenated, fed to Script 06
```

> `full_training_table.csv` is the only file Script 06 needs. It contains 58,214 rows across 33 matches, with zero event-derived columns.

---

### Step 6 — Train Classifiers

**What it does:** Loads `full_training_table.csv`, splits by match (never by row), trains two models — a binary PASS classifier and a 12-class multiclass classifier — using 5-fold GroupKFold cross-validation grouped by match ID, evaluates on held-out test matches, and saves all results.

**Stage 1:** Binary PASS vs NOT-PASS using XGBoost with `scale_pos_weight` for class imbalance.

**Stage 2:** 12-class multiclass using XGBoost with `sample_weight`. Results reported twice — on the full 12-class label, and on grouped classes matching Script 04b's structure — so the ML model and rule baseline can be compared fairly.

Missing values (nulls) in feature columns are handled natively by XGBoost. No imputation is applied.

**Run:**

```bash
python3 06_train_classifier_v7.py
```

**Full argument list (all optional):**

```
--features    Path to full_training_table.csv    default: ./results/features_v7/full_training_table.csv
--out         Where to save model results         default: ./results/models_v7
--test-size   Fraction of matches held out        default: 0.21 (~7 of 33 matches)
--cv-folds    Number of GroupKFold folds          default: 5
--seed        Random seed for reproducibility     default: 42
--min-class   Min training rows to keep a class   default: 10
```

**Produces in `results/models_v7/`:**

```
stage1_pass_binary.json
stage1_feature_importance.csv
stage2_multiclass.json
stage2_feature_importance.csv
stage2_confusion_matrix_full.csv
stage2_confusion_matrix_grouped.csv
```

**To save the full console output to a log file:**

```bash
python3 06_train_classifier_v7.py | tee results/models_v7/run_log.txt
```

---

## What Gets Produced

Running the full pipeline produces the following in your `results/` folder:

```
results/
├── eda_output/
│   ├── eda_summary.txt                    <- full text log of all EDA findings
│   └── *.png                              <- 8 analysis plots
├── aligned/
│   ├── <match_id>_tracking_clean.csv      <- cleaned tracking, one per match
│   ├── <match_id>_events_aligned.csv      <- events in physical coordinates
│   └── <match_id>_match_meta.json         <- team IDs, GK IDs, FPS
├── joined/
│   └── <match_id>_training_data.csv       <- tracking + events joined per frame
├── labelled/
│   └── <match_id>_labelled.csv            <- every frame with its event label
├── rule_baseline/
│   ├── rule_predictions.csv               <- predicted vs true label per anchor
│   ├── rule_baseline.json                 <- per-class P/R/F1, macro F1 = 0.188
│   └── rule_baseline_confusion_matrix.csv
├── features_v7/
│   ├── <match_id>_features.csv            <- one row per anchor per match
│   └── full_training_table.csv            <- 58,214 rows, all 33 matches combined
└── models_v7/
    ├── stage1_pass_binary.json            <- binary PASS CV + test results
    ├── stage1_feature_importance.csv
    ├── stage2_multiclass.json             <- multiclass CV + test results (full + grouped)
    ├── stage2_feature_importance.csv
    ├── stage2_confusion_matrix_full.csv   <- 13x13 true vs predicted
    └── stage2_confusion_matrix_grouped.csv <- 6x6 grouped vs rule baseline
```

---

## The 5 Engineered Features

All features are computed purely from tracking data. None reference event annotation columns.

**1. Ball speed** — frame-to-frame distance divided by 0.1 seconds (10Hz), computed on consecutive non-interpolated frames, capped at 60 m/s. Mean, max, and value at anchor frame.

**2. Ball height** — read directly from `ball_z` (metres). Max, mean, and count of frames above 1.5m across the window.

**3. Ball-to-goal distance** — Euclidean distance from `ball_x`/`ball_y` at anchor to each goal mouth (home goal at x=0, away goal at x=105, both at y=34). Plus pitch third: 0 = defensive (x<35), 1 = midfield (35≤x<70), 2 = attacking (x≥70).

**4. Possession change** — nearest-player-to-ball team at window start vs window end. `1` if team flips, `0` if same team, `null` if ball or players not tracked at either boundary.

**5. Opponent proximity** — at the anchor frame, count of opposing players within 1.5m and 3m of the ball. "Opposing" means the team not currently closest to the ball.

---

## Target Event Classes

**12 classes the model detects:**

```
PASS            TACKLE          INTERCEPTION    CLEARANCE
BALL RECOVERY   AERIAL          SAVE            MISSED SHOT
SAVED SHOT      GOAL            DISPOSSESSED    KEEPER PICKUP
```

Plus `NO EVENT` as the background class — 13 classes total for the multiclass model.

**Excluded from detection** (not distinguishable from x/y tracking alone):

| Class | Reason |
|---|---|
| FOUL | No contact signal from (x,y) dots — needs pose or video |
| BALL TOUCH | Too fine-grained for 10Hz — indistinguishable from carry |
| TAKEON | Dribble attempt — indistinguishable from carry at 10Hz |
| CHALLENGE | Broad overlap with TACKLE — causes label noise |
| CARD | Administrative consequence of FOUL |
| SUBSTITUTION | Administrative — no ball or position signal |
| END | Match boundary marker |
| FORMATION CHANGE | Tactical annotation — no tracking signal |
| CLAIM | Goalkeeper sub-event — covered by SAVE |
| OFFSIDE PASS | Requires offside line computation — out of scope |
| CHANCE MISSED | Editorial label — not independently detectable |

---

## Results

### Stage 1 — Binary PASS classifier

| Metric | Score |
|---|---|
| CV macro F1 | 0.628 (± 0.028 across 5 folds) |
| Test macro F1 | 0.654 |
| PASS precision / recall | 0.67 / 0.72 |
| NOT-PASS precision / recall | 0.64 / 0.59 |

### Stage 2 — Multiclass grouped comparison vs rule baseline

| Class | ML model F1 | Rule baseline F1 |
|---|---|---|
| NO EVENT | 0.51 | 0.38 |
| PASS | 0.59 | 0.25 |
| AERIAL | 0.16 | 0.06 |
| POSSESSION_CHANGE_GROUP | 0.15 | 0.08 |
| SHOT_OR_GK_GROUP | 0.10 | 0.005 |
| CLEARANCE | 0.00 | 0.01 |
| **Macro F1 (overall)** | **0.253** | **0.188** |

The ML model outperforms the rule-based baseline across all classes except CLEARANCE, where both essentially fail due to very low test support (57 rows).

### Reading the result files

**JSON files** (`stage1_pass_binary.json`, `stage2_multiclass.json`, `rule_baseline.json`):
- `cv.fold_f1` — list of macro F1 scores, one per cross-validation fold
- `cv.mean_f1` / `cv.std_f1` — average and spread across folds
- `test_report` / `test_report_full` — per-class precision, recall, F1, support on held-out test matches
- `macro_f1_full` / `macro_f1_grouped` — headline macro F1 scores
- `confusion_matrix` — Stage 1 only, 2x2 grid in order [NOT-PASS, PASS]

**Confusion matrix CSVs**: rows = true label, columns = predicted label. The diagonal is correct predictions; everything off-diagonal is a specific type of mistake.

**Feature importance CSVs**: features ranked by contribution to model decisions. Scores are relative (sum to ~1 within each file), not percentages of accuracy.

---

## Known Limitations

- **TACKLE / INTERCEPTION / BALL RECOVERY / DISPOSSESSED** score F1 0.03–0.07 individually — these events are physically ambiguous at 10Hz without pose or video
- **GOAL, KEEPER PICKUP, CLEARANCE** have very few test rows (8–57), making F1 unstable for those classes
- **`possession_change_in_window`** is null for ~45% of rows due to ball tracking gaps around high-action moments — handled natively by XGBoost, not imputed
- **Null rates are higher for rare classes** (GOAL: 17%, MISSED SHOT: 18%) because those events happen in the most visually chaotic moments where tracking is most likely to drop out — this is expected and confirmed in the data
- **Match 745399** is included in all runs pending mentor confirmation on a reported duplication issue

---

## Authors

IE Business School · Driblab Capstone Group · 2026
