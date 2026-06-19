"""
=============================================================
DRIBLAB CAPSTONE 2026 — STEP 6: Train event classifiers (v7)
=============================================================

Input : full_training_table.csv  (Step 5 v7 output — 58,214 rows, 33 matches,
        12 tracking-only feature columns, zero event-derived columns)
Output: model results, per-class metrics, feature importance, confusion
        matrices, saved under --out

SCOPE FOR THIS ITERATION
-------------------------
Per the mentor's instruction, this run uses exactly the 5 feature groups
already agreed and locked in, no additive feature loop this pass:

  1. Ball speed            -> ball_speed_mean, ball_speed_max, ball_speed_at_anchor
  2. Ball height            -> ball_z_max, ball_z_mean, frames_z_above_1_5m
  3. Ball-to-goal distance  -> ball_dist_to_home_goal, ball_dist_to_away_goal,
                               ball_pitch_third
  4. Possession change      -> possession_change_in_window
  5. Opponent proximity     -> n_opponents_within_1m5, n_opponents_within_3m

  + period (match context, not a chosen feature group, included as it
    carries no leak risk — same half-by-half framing 04b already uses)

Explicitly EXCLUDED from the model inputs:
  ball_available_ratio, cam_present_ratio
    -> data-quality diagnostics, not football signal. Including them
       risks the model learning "tracking is unreliable here, guess
       NO EVENT" rather than learning real ball/player movement. Kept
       in the table for audit purposes only, never passed to the model.
  match_id, anchor_frame, label
    -> identifiers / target, not features.

No event-derived column exists anywhere in the input table (verified
upstream in Step 5 with a hard assertion). This script adds a second,
independent guard below so a future schema change cannot silently
reintroduce one without the script failing loudly.

STRUCTURE (unchanged from v6, since this part already worked correctly)
--------------------------------------------------------------------
1. Load the single concatenated table (already produced by Step 5 — this
   script does not re-concatenate per-match files).
2. Match-level train/test split — never split by row. ~20-30% of the 33
   matches held out completely; all rows from each match stay together.
3. Stage 1 — Binary: PASS vs NOT-PASS.
     5-fold GroupKFold CV (groups=match_id) on the training matches.
     XGBoost, scale_pos_weight for class imbalance.
     NaNs handled natively by XGBoost — no imputation (see note below).
4. Stage 2 — Multiclass: all 12 target classes + NO EVENT.
     Same CV scheme, sample_weight for class imbalance.
     Reported twice: full 12-class P/R/F1, and a grouped view that
     collapses TACKLE/INTERCEPTION/BALL RECOVERY/DISPOSSESSED into
     POSSESSION_CHANGE_GROUP and SAVE/SAVED SHOT/MISSED SHOT/GOAL/
     KEEPER PICKUP into SHOT_OR_GK_GROUP — the same groups
     04b_rule_based_detector.py uses — so the ML-vs-rules comparison is
     apples to apples. This grouping is applied ONLY at evaluation time;
     the model is trained on the full 12-class label, never on the
     grouped label.
5. Evaluation is by per-class precision/recall/F1 throughout. Accuracy is
   never used as a decision metric (NO EVENT dominates ~29% of rows,
   PASS ~57%; accuracy would reward predicting the majority class).

On missing values
------------------
Several features (especially possession_change_in_window, which needs
a tracked player from both teams at both window-start and window-end)
are null on a meaningful fraction of rows — this reflects genuine
tracking-camera gaps (documented in the kickoff guide: ball present
roughly two-thirds of all frames), not a bug. XGBoost learns, at each
tree split, which branch a missing value should default to, based on
what minimizes training loss — this is the same approach the original
v6 trainer used and it requires no imputation step.

Usage
-----
  python 06_train_classifier_v7.py --features ./features_v7/full_training_table.csv --out ./models_v7
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── feature set — the 5 agreed groups, 12 columns + period ────────────────
FEATURES = [
    # 1. Ball speed
    "ball_speed_mean", "ball_speed_max", "ball_speed_at_anchor",
    # 2. Ball height
    "ball_z_max", "ball_z_mean", "frames_z_above_1_5m",
    # 3. Ball-to-goal distance (from ball_x/ball_y, never event_x_m/y_m)
    "ball_dist_to_home_goal", "ball_dist_to_away_goal", "ball_pitch_third",
    # 4. Possession change in window
    "possession_change_in_window",
    # 5. Opponent proximity
    "n_opponents_within_1m5", "n_opponents_within_3m",
    # match context, not a chosen feature group, included as-is
    "period",
]

# Explicitly never used as model inputs — diagnostics only.
EXCLUDED_DIAGNOSTIC_COLS = {"ball_available_ratio", "cam_present_ratio"}

# Defensive guard — if any of these ever appear in FEATURES, fail loudly
# rather than silently retraining on a leaked column.
FORBIDDEN_COLS = {
    "x_m", "y_m", "x_end_m", "y_end_m", "outcome", "attack_dir",
    "event_id", "event_abs_sec", "event_dist_to_home_goal",
    "event_dist_to_away_goal", "event_x_m", "event_y_m", "event_x_end_m",
    "event_y_end_m", "event_outcome", "pass_length", "min_dist_home_to_ball",
    "min_dist_away_to_ball", "gk_defending_dist", "gk_attacking_dist",
    "gk_in_goal_area",
}
assert not (FORBIDDEN_COLS & set(FEATURES)), \
    f"LEAKAGE GUARD TRIPPED: forbidden columns found in FEATURES: {FORBIDDEN_COLS & set(FEATURES)}"

TARGET_EVENTS = [
    "NO EVENT", "PASS", "TACKLE", "INTERCEPTION", "CLEARANCE",
    "BALL RECOVERY", "AERIAL", "SAVE", "MISSED SHOT", "SAVED SHOT",
    "GOAL", "DISPOSSESSED", "KEEPER PICKUP",
]

# Same collapse groups as 04b_rule_based_detector.py — evaluation only.
POSSESSION_GROUP = {"TACKLE", "INTERCEPTION", "BALL RECOVERY", "DISPOSSESSED"}
SHOT_GROUP       = {"SAVE", "SAVED SHOT", "MISSED SHOT", "GOAL", "KEEPER PICKUP"}


def collapse_to_group(label: str) -> str:
    if label in SHOT_GROUP:
        return "SHOT_OR_GK_GROUP"
    if label in POSSESSION_GROUP:
        return "POSSESSION_CHANGE_GROUP"
    return label


# ── helpers ─────────────────────────────────────────────────────────────

def load_table(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path)

    present_forbidden = FORBIDDEN_COLS & set(df.columns)
    if present_forbidden:
        raise ValueError(
            f"LEAKAGE GUARD TRIPPED: input table contains forbidden columns "
            f"{present_forbidden}. Refusing to train. Re-check Step 5 output."
        )

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Expected feature columns missing from input table: {missing_features}")

    print(f"Loaded {features_path}: {len(df)} rows, {df['match_id'].nunique()} matches")
    print(df["label"].value_counts().to_string())
    return df


def split_by_match(df: pd.DataFrame, test_size: float, seed: int = 42) -> tuple:
    """Split by match_id, never by row — all windows from a given match
    stay entirely in train or entirely in test."""
    match_ids = sorted(df["match_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(match_ids)

    n_test = max(1, round(len(match_ids) * test_size))
    n_test = min(n_test, len(match_ids) - 1)
    test_ids  = set(match_ids[:n_test])
    train_ids = set(match_ids[n_test:])

    train_df = df[df["match_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["match_id"].isin(test_ids)].reset_index(drop=True)

    print(f"\nTrain: {len(train_ids)} matches ({len(train_ids)/len(match_ids):.1%}), {len(train_df)} rows")
    print(f"Test:  {len(test_ids)} matches ({len(test_ids)/len(match_ids):.1%}), {len(test_df)} rows")
    print(f"Test match IDs: {sorted(test_ids)}")
    return train_df, test_df


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts  = y.value_counts()
    n_total = len(y)
    n_class = len(counts)
    weights = {cls: n_total / (n_class * cnt) for cls, cnt in counts.items()}
    return y.map(weights).values


def cv_evaluate(model, X, y, groups, n_splits, multiclass: bool) -> dict:
    n_splits = max(2, min(n_splits, groups.nunique()))
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1 = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m = model.__class__(**model.get_params())
        if multiclass:
            sw = compute_sample_weights(y_tr)
            m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            m.set_params(scale_pos_weight=spw)
            m.fit(X_tr, y_tr)

        pred = m.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        fold_f1.append(f1)
        print(f"  Fold {fold}: macro F1 = {f1:.4f}  (train matches={tr_idx.size}, val matches={groups.iloc[val_idx].nunique()})")

    print(f"  CV mean macro F1 = {np.mean(fold_f1):.4f} (+/- {np.std(fold_f1):.4f})")
    return {"fold_f1": fold_f1, "mean_f1": float(np.mean(fold_f1)), "std_f1": float(np.std(fold_f1))}


def save_json(obj: dict, out_dir: Path, name: str) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


# ── Stage 1: binary PASS classifier ────────────────────────────────────

def stage1_binary(train_df, test_df, out_dir, n_splits):
    print("\n" + "=" * 60)
    print("STAGE 1: Binary classifier — PASS vs NOT-PASS")
    print("=" * 60)

    X_train = train_df[FEATURES]
    y_train = (train_df["label"] == "PASS").astype(int)
    groups  = train_df["match_id"]

    X_test = test_df[FEATURES]
    y_test = (test_df["label"] == "PASS").astype(int)

    print(f"\nTrain: {len(X_train)} rows (PASS={y_train.sum()}, NOT={(~y_train.astype(bool)).sum()})")
    print(f"Test:  {len(X_test)} rows (PASS={y_test.sum()}, NOT={(~y_test.astype(bool)).sum()})")

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {spw:.3f}")

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )

    print(f"\n--- GroupKFold CV (groups=match_id) ---")
    cv_results = cv_evaluate(model, X_train, y_train, groups, n_splits, multiclass=False)

    print("\n--- Training on full training set ---")
    model.fit(X_train, y_train)

    print("\n--- Held-out test set evaluation ---")
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, target_names=["NOT-PASS", "PASS"],
                                    zero_division=0, output_dict=True)
    print(classification_report(y_test, pred, target_names=["NOT-PASS", "PASS"], zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print("Confusion matrix [rows=true, cols=pred], order=[NOT-PASS, PASS]:")
    print(cm)

    fi = pd.DataFrame({
        "feature": FEATURES, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(fi.to_string(index=False))

    save_json({"cv": cv_results, "test_report": report, "confusion_matrix": cm.tolist()},
              out_dir, "stage1_pass_binary")
    fi.to_csv(out_dir / "stage1_feature_importance.csv", index=False)
    return model


# ── Stage 2: multiclass classifier ─────────────────────────────────────

def stage2_multiclass(train_df, test_df, out_dir, n_splits, min_class_count=10):
    print("\n" + "=" * 60)
    print("STAGE 2: Multiclass classifier — 12 target classes + NO EVENT")
    print("=" * 60)

    label_counts = train_df["label"].value_counts()
    keep_labels  = label_counts[label_counts >= min_class_count].index.tolist()
    dropped      = label_counts[label_counts < min_class_count].index.tolist()
    if dropped:
        print(f"\nDropping under-represented classes (< {min_class_count} train rows): {dropped}")

    train_df = train_df[train_df["label"].isin(keep_labels)].copy()
    test_df  = test_df[test_df["label"].isin(keep_labels)].copy()

    le = LabelEncoder()
    le.fit(sorted(keep_labels))
    y_train = pd.Series(le.transform(train_df["label"]), index=train_df.index)
    y_test  = pd.Series(le.transform(test_df["label"]), index=test_df.index)
    groups  = train_df["match_id"]

    X_train = train_df[FEATURES]
    X_test  = test_df[FEATURES]

    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        num_class=len(le.classes_), objective="multi:softmax",
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )

    print(f"\n--- GroupKFold CV (groups=match_id) ---")
    cv_results = cv_evaluate(model, X_train, y_train, groups, n_splits, multiclass=True)

    print("\n--- Training on full training set ---")
    sw_train = compute_sample_weights(train_df["label"])
    model.fit(X_train, y_train, sample_weight=sw_train)

    print("\n--- Held-out test set evaluation (full 12-class label) ---")
    pred = model.predict(X_test)
    pred_labels = le.inverse_transform(pred)
    true_labels = le.inverse_transform(y_test)

    full_report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
    print(classification_report(true_labels, pred_labels, zero_division=0))

    macro_f1_full = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    print(f"Macro F1 (full 12-class): {macro_f1_full:.4f}")

    cm_full = confusion_matrix(true_labels, pred_labels, labels=sorted(keep_labels))
    pd.DataFrame(cm_full, index=sorted(keep_labels), columns=sorted(keep_labels)).to_csv(
        out_dir / "stage2_confusion_matrix_full.csv"
    )

    # ── grouped evaluation, matching 04b's collapsed classes ───────────
    print("\n--- Held-out test set evaluation (grouped, vs 04b rule baseline) ---")
    true_grouped = [collapse_to_group(l) for l in true_labels]
    pred_grouped = [collapse_to_group(l) for l in pred_labels]
    grouped_labels = sorted(set(true_grouped) | set(pred_grouped))

    grouped_report = classification_report(true_grouped, pred_grouped, labels=grouped_labels,
                                             zero_division=0, output_dict=True)
    print(classification_report(true_grouped, pred_grouped, labels=grouped_labels, zero_division=0))

    macro_f1_grouped = f1_score(true_grouped, pred_grouped, labels=grouped_labels,
                                 average="macro", zero_division=0)
    print(f"Macro F1 (grouped, vs 04b): {macro_f1_grouped:.4f}")

    cm_grouped = confusion_matrix(true_grouped, pred_grouped, labels=grouped_labels)
    pd.DataFrame(cm_grouped, index=grouped_labels, columns=grouped_labels).to_csv(
        out_dir / "stage2_confusion_matrix_grouped.csv"
    )

    fi = pd.DataFrame({
        "feature": FEATURES, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(fi.to_string(index=False))

    save_json({
        "cv": cv_results,
        "test_report_full": full_report,
        "macro_f1_full": macro_f1_full,
        "test_report_grouped": grouped_report,
        "macro_f1_grouped": macro_f1_grouped,
        "classes": list(le.classes_),
    }, out_dir, "stage2_multiclass")
    fi.to_csv(out_dir / "stage2_feature_importance.csv", index=False)

    return model, macro_f1_grouped


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Capstone Step 6 v7 — train classifiers on leak-free features")
    ap.add_argument("--features",  default="./results/features_v7/full_training_table.csv",
                    help="Path to the single concatenated features CSV from Step 5")
    ap.add_argument("--out",       default="./results/models_v7")
    ap.add_argument("--test-size", type=float, default=0.21,
                    help="Fraction of matches held out for test (default 0.21, ~7 of 33)")
    ap.add_argument("--cv-folds",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--min-class", type=int, default=10)
    args = ap.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: Load concatenated feature table ===")
    df = load_table(features_path)

    print("\n=== STEP 2: Match-level train/test split ===")
    train_df, test_df = split_by_match(df, args.test_size, args.seed)

    print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")
    print(f"Diagnostic columns present but excluded from model: {sorted(EXCLUDED_DIAGNOSTIC_COLS)}")

    stage1_binary(train_df, test_df, out_dir, n_splits=args.cv_folds)
    _, rule_comparison_f1 = stage2_multiclass(train_df, test_df, out_dir,
                                               n_splits=args.cv_folds,
                                               min_class_count=args.min_class)

    print("\n=== DONE ===")
    print(f"All results saved to {out_dir}/")
    print(f"\nStage 2 macro F1 (grouped, vs 04b rule baseline) = {rule_comparison_f1:.4f}")
    print("Compare this directly against rule_baseline.json's macro_f1 from 04b_rule_based_detector.py")
    print("to see how much the ML model improves on the rule-based floor.")


if __name__ == "__main__":
    main()
