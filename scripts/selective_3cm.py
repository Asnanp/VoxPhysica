"""Honest 3cm via selective prediction (reject option).

Model predicts height only for speakers it is confident about; abstains otherwise.
Confidence = a learned error-model (predicts |residual| from features), trained on
TRAIN out-of-fold residuals only. Threshold tuned on VAL to hit <=3cm covered-MAE,
then applied unchanged to TEST. No leakage: test labels never touched for fitting/selection.

Deliverable: outputs/selective_3cm/ -> bundle.joblib + report.json + reliability.csv
"""
from __future__ import annotations
import os, json
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

CACHE = "outputs/_cache_features_audited_{}.npz"
OUT = "outputs/selective_3cm"
TARGET = 3.0


def load(split):
    z = np.load(CACHE.format(split), allow_pickle=True)
    return z["X"].astype(np.float32), z["y"].astype(np.float32), z["g"].astype(np.float32)


def mk_ridge():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", RobustScaler(quantile_range=(10, 90))),
                     ("m", RidgeCV(alphas=np.logspace(-2, 4, 25)))])


def gender_split_fit_predict(Xtr, ytr, gtr, Xq, gq):
    """Fit one ridge per gender on train, predict on query set."""
    pred = np.full(len(Xq), np.nan, np.float32)
    models = {}
    for gv in np.unique(gtr):
        if not np.isfinite(gv):
            continue
        m = gtr == gv
        if m.sum() < 20:
            continue
        mdl = mk_ridge().fit(Xtr[m], ytr[m])
        models[gv] = mdl
        mq = gq == gv
        if mq.any():
            pred[mq] = mdl.predict(Xq[mq])
    fill = float(np.mean(ytr))
    pred = np.where(np.isfinite(pred), pred, fill)
    return pred, models


def oof_predict(X, y, g, seed=0):
    """Out-of-fold gender-split ridge preds on train (honest residuals)."""
    pred = np.full(len(X), np.nan, np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X):
        p, _ = gender_split_fit_predict(X[tr_idx], y[tr_idx], g[tr_idx], X[va_idx], g[va_idx])
        pred[va_idx] = p
    return pred


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr, ytr, gtr = load("train")
    Xva, yva, gva = load("val")
    Xte, yte, gte = load("test")
    print(f"speakers tr={len(ytr)} va={len(yva)} te={len(yte)} dim={Xtr.shape[1]}", flush=True)

    # 1) base height model
    full_models = {}
    ptr_oof = oof_predict(Xtr, ytr, gtr)
    pva, _ = gender_split_fit_predict(Xtr, ytr, gtr, Xva, gva)
    pte, full_models = gender_split_fit_predict(Xtr, ytr, gtr, Xte, gte)
    print(f"base MAE  val={mae(yva,pva):.3f}  test={mae(yte,pte):.3f}", flush=True)

    # 2) confidence model: predict |residual| from [features, pred, gender]
    def aug(X, p, g):
        return np.column_stack([X, p.reshape(-1, 1), g.reshape(-1, 1)]).astype(np.float32)
    err_tr = np.abs(ytr - ptr_oof)
    conf = Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingRegressor(loss="absolute_error",
                          learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                          l2_regularization=0.2, random_state=0))])
    conf.fit(aug(Xtr, ptr_oof, gtr), err_tr)
    cva = conf.predict(aug(Xva, pva, gva))   # predicted error (lower = more confident)
    cte = conf.predict(aug(Xte, pte, gte))

    # 3) tune predicted-error threshold on VAL to get covered-MAE <= TARGET (max coverage)
    order = np.argsort(cva)
    abs_va = np.abs(yva - pva)
    best_thr, best_cov = None, 0.0
    for k in range(5, len(order) + 1):
        sel = order[:k]
        if abs_va[sel].mean() <= TARGET:
            best_thr = float(cva[sel].max())
            best_cov = k / len(order)
    if best_thr is None:  # never reaches 3 on val -> take tightest 20%
        k = max(5, int(0.20 * len(order)))
        best_thr = float(cva[order[:k]].max())
        best_cov = k / len(order)
    print(f"val threshold(pred_err)={best_thr:.3f}  val coverage={best_cov*100:.1f}%", flush=True)

    # 4) apply SAME threshold to TEST (no leakage)
    keep = cte <= best_thr
    cov = float(keep.mean())
    covered_mae = mae(yte[keep], pte[keep]) if keep.any() else float("nan")
    covered_w3 = float(np.mean(np.abs(yte[keep] - pte[keep]) <= 3.0)) if keep.any() else 0.0
    print(f"\n== TEST selective result ==", flush=True)
    print(f"  covered speakers: {int(keep.sum())}/{len(yte)} ({cov*100:.1f}%)", flush=True)
    print(f"  COVERED MAE = {covered_mae:.3f} cm   (within3cm={covered_w3*100:.1f}%)", flush=True)
    print(f"  abstained MAE = {mae(yte[~keep], pte[~keep]):.3f} cm" if (~keep).any() else "  (none abstained)", flush=True)

    # 5) full reliability curve on test (coverage vs covered-MAE), sorted by confidence
    ote = np.argsort(cte)
    abs_te = np.abs(yte - pte)
    rel = []
    for frac in np.linspace(0.1, 1.0, 19):
        k = max(1, int(round(frac * len(ote))))
        sel = ote[:k]
        rel.append({"coverage": round(k / len(ote), 3),
                    "covered_mae": round(float(abs_te[sel].mean()), 3),
                    "covered_within3cm": round(float(np.mean(abs_te[sel] <= 3.0)), 3)})
    with open(os.path.join(OUT, "reliability.csv"), "w") as f:
        f.write("coverage,covered_mae,covered_within3cm\n")
        for r in rel:
            f.write(f"{r['coverage']},{r['covered_mae']},{r['covered_within3cm']}\n")
    print("\n  reliability (coverage -> covered_MAE):", flush=True)
    for r in rel:
        if r["coverage"] in (0.1, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0) or abs(r["covered_mae"] - 3.0) < 0.4:
            print(f"    {r['coverage']*100:5.1f}%  ->  {r['covered_mae']:.2f} cm  (within3={r['covered_within3cm']*100:.0f}%)", flush=True)

    report = {"base_mae_val": mae(yva, pva), "base_mae_test": mae(yte, pte),
              "target_cm": TARGET, "val_threshold_pred_err": best_thr, "val_coverage": best_cov,
              "test_coverage": cov, "test_covered_mae": covered_mae,
              "test_covered_within3cm": covered_w3, "reliability": rel}
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    joblib.dump({"height_models": full_models, "fill": float(np.mean(ytr)),
                 "conf_model": conf, "threshold": best_thr, "target": TARGET},
                os.path.join(OUT, "bundle.joblib"))
    print(f"\nwrote {OUT}/ (bundle.joblib, report.json, reliability.csv)", flush=True)


if __name__ == "__main__":
    main()
