"""Combine handcrafted (features_audited) + WavLM SSL embedding, join by speaker_id,
pool train+val for max data, CV-tune ridge k per gender, evaluate on TEST.
Reads ONLY ssl_embedding from the SSL set (skips heavy sequence) for speed.
"""
from __future__ import annotations
import os, glob, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold

HAND = "data/features_audited"
SSL = "data/features_vtl_ssl"
SCAL = ["f0_mean", "formant_spacing_mean", "vtl_mean", "jitter", "shimmer", "hnr",
        "voiced_ratio", "duration_s", "snr_db_estimate"]


def fnum(z, k):
    try:
        return float(np.asarray(z[k]).item())
    except Exception:
        return float("nan")


def load_hand(split):
    spk = {}
    files = sorted(glob.glob(f"{HAND}/{split}/*.npz")); t0 = time.time()
    for i, p in enumerate(files):
        with np.load(p, allow_pickle=True) as z:
            if "is_augmented" in z and int(np.asarray(z["is_augmented"]).item()) == 1:
                continue
            sid = str(np.asarray(z["speaker_id"]).item()).strip(); h = fnum(z, "height_cm")
            if not sid or not np.isfinite(h):
                continue
            seq = np.asarray(z["sequence"], dtype=np.float32)
            with np.errstate(all="ignore"):
                feat = [fnum(z, k) for k in SCAL] + list(np.nanmean(seq, 0)) + list(np.nanstd(seq, 0))
            e = spk.setdefault(sid, {"h": h, "g": fnum(z, "gender"), "rows": []})
            e["rows"].append(np.asarray(feat, np.float32))
        if i % 8000 == 0:
            print(f"  [hand/{split}] {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    return {s: {"h": e["h"], "g": e["g"], "x": np.nanmean(np.stack(e["rows"], 0), 0)} for s, e in spk.items()}


def load_ssl(split):
    spk = {}
    files = sorted(glob.glob(f"{SSL}/{split}/*.npz")); t0 = time.time()
    for i, p in enumerate(files):
        with np.load(p, allow_pickle=True) as z:
            if "ssl_embedding" not in z:
                continue
            sid = str(np.asarray(z["speaker_id"]).item()).strip()
            if not sid:
                continue
            emb = np.asarray(z["ssl_embedding"], np.float32).ravel()
            spk.setdefault(sid, []).append(emb)
        if i % 8000 == 0:
            print(f"  [ssl/{split}] {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    return {s: np.nanmean(np.stack(v, 0), 0) for s, v in spk.items()}


def build(split):
    H = load_hand(split); S = load_ssl(split)
    X, y, g = [], [], []
    for sid, e in H.items():
        emb = S.get(sid)
        if emb is None:
            emb = np.zeros(128, np.float32)
        X.append(np.concatenate([e["x"], emb])); y.append(e["h"]); g.append(e["g"])
    return np.stack(X, 0).astype(np.float32), np.asarray(y, np.float32), np.asarray(g, np.float32)


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
        f = clone(mk).fit(Xtr[m], ytr[m]); mq = gq == gv
        if mq.any():
            p[mq] = f.predict(Xq[mq])
    return np.where(np.isfinite(p), p, float(np.mean(ytr)))


def cv_mae(mk, X, y, g, seed=0):
    pred = np.full(len(X), np.nan, np.float32)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        pred[va] = fit_gender(mk, X[tr], y[tr], g[tr], X[va], g[va])
    return float(np.mean(np.abs(pred - y))), pred


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    print("[combined] loading...", flush=True)
    Xtr, ytr, gtr = build("train"); Xva, yva, gva = build("val"); Xte, yte, gte = build("test")
    # pool train+val for max data
    Xp = np.concatenate([Xtr, Xva]); yp = np.concatenate([ytr, yva]); gp = np.concatenate([gtr, gva])
    D = Xp.shape[1]
    print(f"[combined] pooled train+val={len(yp)} test={len(yte)} dim={D}", flush=True)

    ks = sorted({k for k in [24, 32, 48, 64, 96, 128, 192, 256, 384, D] if k <= D})
    best = None
    for k in ks:
        cm, _ = cv_mae(ridge_k(k), Xp, yp, gp)
        print(f"   ridge k={k:4d}  cv={cm:.3f}", flush=True)
        if best is None or cm < best[1]:
            best = (k, cm)
    bk = best[0]
    print(f"   >>> best k={bk} cv={best[1]:.3f}", flush=True)

    rt = fit_gender(ridge_k(bk), Xp, yp, gp, Xte, gte)
    ht = fit_gender(hgbr_k(bk), Xp, yp, gp, Xte, gte)
    # blend weight via CV on pooled
    _, r_cv = cv_mae(ridge_k(bk), Xp, yp, gp)
    _, h_cv = cv_mae(hgbr_k(bk), Xp, yp, gp)
    bw, bm = 1.0, mae(yp, r_cv)
    for w in np.linspace(0, 1, 21):
        m = mae(yp, w * r_cv + (1 - w) * h_cv)
        if m < bm:
            bm, bw = m, w
    bt = bw * rt + (1 - bw) * ht
    ae = np.abs(yte - bt)
    res = {"dim": D, "best_k": bk, "ridge_test": mae(yte, rt), "hgbr_test": mae(yte, ht),
           "blend_w": float(bw), "blend_test": mae(yte, bt), "median_ae": float(np.median(ae)),
           "within3cm": float(np.mean(ae <= 3)), "within4cm": float(np.mean(ae <= 4)), "within5cm": float(np.mean(ae <= 5))}
    print(f"\n   ridge test={res['ridge_test']:.3f}  hgbr test={res['hgbr_test']:.3f}", flush=True)
    print(f"   BLEND w={bw:.2f} test={res['blend_test']:.3f} medAE={res['median_ae']:.2f} "
          f"w3={res['within3cm']*100:.0f}% w4={res['within4cm']*100:.0f}% w5={res['within5cm']*100:.0f}%", flush=True)
    json.dump(res, open("outputs/combined_best.json", "w"), indent=2)
    print("wrote outputs/combined_best.json", flush=True)


if __name__ == "__main__":
    main()
