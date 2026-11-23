"""Train 30/90/365-day asset failure-risk models and write honest evaluation metrics.

Evaluation protocol
-------------------
* Monthly snapshots; features use only data before each snapshot.
* Time-based split: the last 12 snapshots with fully observed labels are the
  test set. For horizon h, training snapshots must satisfy
  snapshot + h <= test_start so no training label peeks into the test period
  ("purged" split).
* Baselines: (1) rank by the asset's own incident count over the last year,
  (2) the same Random Forest trained on only the ~14 features the original
  hackathon model used. The gap shows what the feature engineering adds.

Usage: python -m ml.risk.train
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from app.config import ARTIFACTS_DIR, get_settings
from ml.risk.data import load_context
from ml.risk.features import HORIZONS

LEGACY_FEATURES = [  # closest equivalents of the original model's real (non one-hot) inputs
    "wx_precip_7d", "wx_snow_7d", "wx_freeze_thaw_7d", "wx_temp_mean_7d", "x_km", "y_km",
    "nbr_inc_100m_365d", "nbr_sev_250m_365d", "rep_sev_100m_90d", "rep_100m_30d",
    "own_inc_total", "own_days_since_inc", "own_inc_30d", "is_road",
]


def make_model() -> RandomForestClassifier:
    # max_samples: each tree sees a 20% bootstrap sample -> ~5x faster with negligible accuracy loss
    return RandomForestClassifier(
        n_estimators=150, min_samples_leaf=50, max_features="sqrt", max_samples=0.2,
        n_jobs=-1, random_state=42,
    )


def top_k_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    order = np.argsort(-p)
    out = {}
    for frac in (0.05, 0.10):
        k = max(1, int(len(p) * frac))
        hits = y[order[:k]].sum()
        out[f"precision_at_top{int(frac * 100)}pct"] = float(hits / k)
        out[f"recall_at_top{int(frac * 100)}pct"] = float(hits / max(y.sum(), 1))
    return out


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        **top_k_metrics(y, p),
    }


def main() -> None:
    t0 = time.perf_counter()
    with psycopg.connect(get_settings().database_url) as conn:
        ctx = load_context(conn)
    load_s = time.perf_counter() - t0

    data_end = pd.Timestamp(np.nanmax(ctx.inc_t), unit="D").normalize()
    last_full = data_end - pd.Timedelta(days=max(HORIZONS))
    snapshots = list(pd.date_range("2021-01-01", last_full, freq="MS"))
    test_snaps = snapshots[-12:]
    test_start = test_snaps[0]

    t1 = time.perf_counter()
    frames = {}
    for s in snapshots:
        frames[s] = (ctx.build(s), ctx.labels(s))
    print(f"Built features for {len(snapshots)} snapshots x {ctx.n} assets", flush=True)
    feat_s = time.perf_counter() - t1
    features = list(next(iter(frames.values()))[0].columns)

    X_test = pd.concat([frames[s][0] for s in test_snaps], ignore_index=True)
    Y_test = pd.concat([frames[s][1] for s in test_snaps], ignore_index=True)

    models, metrics = {}, {}
    for h in HORIZONS:
        label = f"fail_in_{h}d"
        train_snaps = [s for s in snapshots if s + pd.Timedelta(days=h) <= test_start]
        X = pd.concat([frames[s][0] for s in train_snaps], ignore_index=True)
        y = pd.concat([frames[s][1] for s in train_snaps], ignore_index=True)[label].to_numpy()
        yt = Y_test[label].to_numpy()

        clf = make_model().fit(X[features], y)
        p = clf.predict_proba(X_test[features])[:, 1]

        legacy = make_model().fit(X[LEGACY_FEATURES], y)
        p_legacy = legacy.predict_proba(X_test[LEGACY_FEATURES])[:, 1]
        p_naive = X_test["own_inc_365d"].to_numpy() + 1e-3 * X_test["nbr_inc_250m_365d"].to_numpy()

        importances = sorted(zip(features, clf.feature_importances_, strict=False), key=lambda t: -t[1])[:15]
        metrics[label] = {
            "train_rows": int(len(y)), "train_snapshots": len(train_snaps),
            "test_rows": int(len(yt)), "test_positive_rate": float(yt.mean()),
            "model": evaluate(yt, p),
            "baseline_legacy_features": evaluate(yt, p_legacy),
            "baseline_own_history": {"roc_auc": float(roc_auc_score(yt, p_naive)),
                                     "pr_auc": float(average_precision_score(yt, p_naive)),
                                     **top_k_metrics(yt, p_naive)},
            "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in importances],
        }
        # Refit on every snapshot with observed labels for production scoring
        Xf = pd.concat([frames[s][0] for s in snapshots], ignore_index=True)
        yf = pd.concat([frames[s][1] for s in snapshots], ignore_index=True)[label].to_numpy()
        models[h] = make_model().fit(Xf[features], yf)
        m = metrics[label]["model"]
        print(f"{label}: ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
              f"recall@top10% {m['recall_at_top10pct']:.1%}  "
              f"(legacy features ROC-AUC {metrics[label]['baseline_legacy_features']['roc_auc']:.3f})")

    version = datetime.now(UTC).strftime("rf-%Y%m%d%H%M")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": version, "features": features, "models": models, "data_end": str(data_end.date())},
                get_settings().risk_model_path, compress=3)
    summary = {
        "version": version,
        "n_features": len(features),
        "n_assets": ctx.n,
        "n_incidents": int(len(ctx.inc_t)),
        "n_reports": int(len(ctx.rep_t)),
        "n_regions": int(len(ctx.region_codes)),
        "snapshots": {"first": str(snapshots[0].date()), "last": str(snapshots[-1].date()),
                      "count": len(snapshots), "test_start": str(test_start.date())},
        "timing_s": {"load": round(load_s, 2), "features_total": round(feat_s, 2),
                     "features_per_snapshot": round(feat_s / len(snapshots), 3)},
        "horizons": metrics,
    }
    (ARTIFACTS_DIR / "risk_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(features)} features, {ctx.n} assets, {len(snapshots)} snapshots; "
          f"features built in {feat_s:.1f}s ({feat_s / len(snapshots):.2f}s/snapshot). Saved {version}.")


if __name__ == "__main__":
    main()
