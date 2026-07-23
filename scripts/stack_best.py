"""Strongest honest stacked height model from a cached speaker matrix.
Per-gender model zoo -> out-of-fold stacking (meta ridge) -> isotonic-free linear calib.
Usage: python scripts/stack_best.py <cache_basename>   e.g. features_vtl_ssl_augmented
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

BASE = sys.argv[1] if len(sys.argv) > 1 else "features_audited"
C = "outputs/_cache_{}_{}.npz"


def load(split):
    z = np.load(C.format(BASE, split), allow_pickle=True)
    return z["X"].astype(np.float32), z["y"].astype(np.float32), z["g"].astype(np.float32)


def zoo(k, seed=0):
    def lin(est):
        return Pipeline([("i", SimpleImputer(strategy="median")), ("s", RobustScaler(quantile_range=(10, 90))),
                         ("k", SelectKBest(f_regression, k=k)), ("m", est)])
    def tre(est):
        return Pipeline([("i", SimpleImputer(strategy="median")), ("k", SelectKBest(f_regression, k=k)), ("m", est)])
    return {
        "ridge": lin(RidgeCV(alphas=np.logspace(-2, 4, 25))),
        "huber": lin(HuberRegressor(alpha=0.003, epsilon=1.35, max_iter=2000)),
        "enet": lin(ElasticNetCV(l1_ratio=(0.1, 0.5, 0.9), cv=5, max_iter=8000, random_state=seed)),
        "extratrees": tre(ExtraTreesRegressor(n_estimators=500, max_features=0.35, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
        "hgbr": tre(HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.03, max_iter=500, max_leaf_nodes=15, l2_regularization=0.1, random_state=seed)),
        "gbr": tre(GradientBoostingRegressor(loss="absolute_error", n_estimators=400, learning_rate=0.03, max_depth=2, subsample=0.85, random_state=seed)),
    }


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def fit_gender(models, Xtr, ytr, gtr, Xq, gq):
    """Fit each model per-gender, return dict name->pred on query."""
    out = {n: np.full(len(Xq), np.nan, np.float32) for n in models}
    for gv in np.unique(gtr):
        if not np.isfinite(gv):
            continue
        m = gtr == gv
        if m.sum() < 25:
            continue
        mq = gq == gv
        for n, mdl in models.items():
            from sklearn.base import clone
            f = clone(mdl).fit(Xtr[m], ytr[m])
            if mq.any():
                out[n][mq] = f.predict(Xq[mq])
    fill = float(np.mean(ytr))
    return {n: np.where(np.isfinite(p), p, fill) for n, p in out.items()}


def oof(models, X, y, g, seed=0):
    out = {n: np.full(len(X), np.nan, np.float32) for n in models}
    kf = KFold(5, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        p = fit_gender(models, X[tr], y[tr], g[tr], X[va], g[va])
        for n in models:
            out[n][va] = p[n]
    return out


def main():
    Xtr, ytr, gtr = load("train")
    Xva, yva, gva = load("val")
    Xte, yte, gte = load("test")
    k = int(min(Xtr.shape[1], max(64, min(400, len(ytr)))))
    print(f"[stack:{BASE}] tr={len(ytr)} va={len(yva)} te={len(yte)} dim={Xtr.shape[1]} k={k}", flush=True)
    Z = zoo(k)

    tr_oof = oof(Z, Xtr, ytr, gtr)
    va_p = fit_gender(Z, Xtr, ytr, gtr, Xva, gva)
    te_p = fit_gender(Z, Xtr, ytr, gtr, Xte, gte)
    for n in Z:
        print(f"   base {n:11s} val={mae(yva,va_p[n]):.3f} test={mae(yte,te_p[n]):.3f}", flush=True)

    names = list(Z)
    Mtr = np.column_stack([tr_oof[n] for n in names])
    Mva = np.column_stack([va_p[n] for n in names])
    Mte = np.column_stack([te_p[n] for n in names])

    # meta-learner: non-negative-ish ridge on OOF base preds, trained on TRAIN oof
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Mtr, ytr)
    sv, st = meta.predict(Mva), meta.predict(Mte)
    # simple val-tuned convex blend as alternative
    from itertools import product
    # calibrate on val (linear)
    cal = LinearRegression().fit(sv.reshape(-1, 1), yva)
    st_c = cal.predict(st.reshape(-1, 1))
    sv_c = cal.predict(sv.reshape(-1, 1))
    res = {
        "base": {n: {"val": mae(yva, va_p[n]), "test": mae(yte, te_p[n])} for n in names},
        "stack_meta": {"val": mae(yva, sv), "test": mae(yte, st), "test_r2": float(r2_score(yte, st))},
        "stack_meta_cal": {"val": mae(yva, sv_c), "test": mae(yte, st_c),
                            "within3cm": float(np.mean(np.abs(yte - st_c) <= 3.0)),
                            "within4cm": float(np.mean(np.abs(yte - st_c) <= 4.0)),
                            "median_ae": float(np.median(np.abs(yte - st_c)))},
    }
    print(f"\n   STACK meta      val={res['stack_meta']['val']:.3f} test={res['stack_meta']['test']:.3f} R2={res['stack_meta']['test_r2']:+.3f}", flush=True)
    print(f"   STACK meta+cal  val={res['stack_meta_cal']['val']:.3f} test={res['stack_meta_cal']['test']:.3f} "
          f"w3={res['stack_meta_cal']['within3cm']*100:.0f}% w4={res['stack_meta_cal']['within4cm']*100:.0f}% medAE={res['stack_meta_cal']['median_ae']:.2f}", flush=True)
    json.dump(res, open(f"outputs/stack_{BASE}.json", "w"), indent=2)
    print(f"\nwrote outputs/stack_{BASE}.json", flush=True)


if __name__ == "__main__":
    main()
