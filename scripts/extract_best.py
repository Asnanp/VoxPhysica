"""Fast signal extraction from cached speaker matrix.
Per-gender ridge with SelectKBest k-sweep (tuned on val) + HGBR + blend + calib.
Usage: python scripts/extract_best.py <cache_basename>
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import clone

BASE = sys.argv[1] if len(sys.argv) > 1 else "features_audited"
C = "outputs/_cache_{}_{}.npz"


def load(s):
    z = np.load(C.format(BASE, s), allow_pickle=True)
    return z["X"].astype(np.float32), z["y"].astype(np.float32), z["g"].astype(np.float32)


def ridge_k(k):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", RobustScaler(quantile_range=(10, 90))),
                     ("k", SelectKBest(f_regression, k=k)), ("m", RidgeCV(alphas=np.logspace(-1, 5, 30)))])


def hgbr_k(k):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("k", SelectKBest(f_regression, k=k)),
                     ("m", HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.04,
                          max_iter=400, max_leaf_nodes=15, l2_regularization=0.2, random_state=0))])


def fit_gender(mk, Xtr, ytr, gtr, Xq, gq):
    p = np.full(len(Xq), np.nan, np.float32)
    for gv in np.unique(gtr):
        if not np.isfinite(gv):
            continue
        m = gtr == gv
        if m.sum() < 25:
            continue
        f = clone(mk).fit(Xtr[m], ytr[m])
        mq = gq == gv
        if mq.any():
            p[mq] = f.predict(Xq[mq])
    return np.where(np.isfinite(p), p, float(np.mean(ytr)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    Xtr, ytr, gtr = load("train"); Xva, yva, gva = load("val"); Xte, yte, gte = load("test")
    D = Xtr.shape[1]
    print(f"[extract:{BASE}] tr={len(ytr)} te={len(yte)} dim={D}", flush=True)
    ks = sorted({k for k in [16, 24, 32, 48, 64, 96, 128, 192, 256, D] if k <= D})

    # ridge k-sweep, pick best on VAL
    best = None
    for k in ks:
        pv = fit_gender(ridge_k(k), Xtr, ytr, gtr, Xva, gva)
        pt = fit_gender(ridge_k(k), Xtr, ytr, gtr, Xte, gte)
        vm, tm = mae(yva, pv), mae(yte, pt)
        print(f"   ridge k={k:4d}  val={vm:.3f}  test={tm:.3f}", flush=True)
        if best is None or vm < best[1]:
            best = (k, vm, tm, pt, pv)
    bk, bvm, btm, ridge_te, ridge_va = best
    print(f"   >>> ridge best k={bk} val={bvm:.3f} test={btm:.3f}", flush=True)

    # hgbr at best k
    hv = fit_gender(hgbr_k(bk), Xtr, ytr, gtr, Xva, gva)
    ht = fit_gender(hgbr_k(bk), Xtr, ytr, gtr, Xte, gte)
    print(f"   hgbr  k={bk}  val={mae(yva,hv):.3f} test={mae(yte,ht):.3f}", flush=True)

    # blend ridge+hgbr, weight tuned on val
    best_w, best_vm = 1.0, mae(yva, ridge_va)
    for w in np.linspace(0, 1, 21):
        vm = mae(yva, w * ridge_va + (1 - w) * hv)
        if vm < best_vm:
            best_vm, best_w = vm, w
    blend_te = best_w * ridge_te + (1 - best_w) * ht
    blend_va = best_w * ridge_va + (1 - best_w) * hv
    # linear calibration on val
    cal = LinearRegression().fit(blend_va.reshape(-1, 1), yva)
    bt = cal.predict(blend_te.reshape(-1, 1))
    ae = np.abs(yte - bt)
    res = {"ridge_best_k": bk, "ridge_test": btm,
           "blend_w_ridge": float(best_w), "blend_val": mae(yva, blend_va),
           "blend_test": mae(yte, blend_te),
           "blend_cal_test": mae(yte, bt), "median_ae": float(np.median(ae)),
           "within3cm": float(np.mean(ae <= 3)), "within4cm": float(np.mean(ae <= 4)),
           "within5cm": float(np.mean(ae <= 5))}
    print(f"\n   BLEND w_ridge={best_w:.2f} val={res['blend_val']:.3f} test={res['blend_test']:.3f}", flush=True)
    print(f"   BLEND+cal test={res['blend_cal_test']:.3f}  medAE={res['median_ae']:.2f}  "
          f"w3={res['within3cm']*100:.0f}% w4={res['within4cm']*100:.0f}% w5={res['within5cm']*100:.0f}%", flush=True)
    json.dump(res, open(f"outputs/extract_{BASE}.json", "w"), indent=2)
    print(f"wrote outputs/extract_{BASE}.json", flush=True)


if __name__ == "__main__":
    main()
