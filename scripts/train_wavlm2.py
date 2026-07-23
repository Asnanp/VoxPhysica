"""Train on improved WavLM-large v2 embeddings. Gender-split ensemble, full-CV.
Optionally fuse handcrafted physics scalars (from seqcache, per-speaker) joined by id.
"""
from __future__ import annotations
import os, json, warnings, collections
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold

W2 = "outputs/wavlm2"
SC = "outputs/seqcache"


def load_emb():
    E, Y, G, ID = [], [], [], []
    for sp in ["train", "val", "test"]:
        z = np.load(f"{W2}/{sp}.npz", allow_pickle=True)
        E.append(z["emb"]); Y.append(z["y"]); G.append(z["g"]); ID.append(z["ids"].astype(str))
    return np.concatenate(E), np.concatenate(Y).astype(np.float32), np.concatenate(G).astype(np.float32), np.concatenate(ID)


def load_physics():
    """per-speaker mean physics scalars from seqcache, keyed by speaker id."""
    out = {}
    for sp in ["train", "val", "test"]:
        z = np.load(f"{SC}/{sp}.npz", allow_pickle=True)
        scal, sid = z["scal"].astype(np.float32), z["sid"].astype(str)
        agg = collections.defaultdict(list)
        for s, v in zip(sid, scal):
            agg[s].append(v)
        for s, v in agg.items():
            out[s] = np.nanmean(np.stack(v), 0)
    return out


def fitg(mk, Xtr, ytr, gtr, Xq, gq):
    p = np.full(len(Xq), np.nan, np.float32)
    for gv in np.unique(gtr):
        m = gtr == gv
        if m.sum() < 20:
            continue
        f = clone(mk).fit(Xtr[m], ytr[m]); mq = gq == gv
        if mq.any():
            p[mq] = f.predict(Xq[mq])
    return np.where(np.isfinite(p), p, float(np.mean(ytr)))


def ridge():
    return Pipeline([("s", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-1, 5, 40)))])
def hgbr():
    return Pipeline([("m", HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.04,
                     max_iter=400, max_leaf_nodes=15, l2_regularization=0.3, random_state=0))])
def etr():
    return Pipeline([("m", ExtraTreesRegressor(n_estimators=400, max_features=0.3, min_samples_leaf=3, n_jobs=-1, random_state=0))])


def cv_blend(X, y, g, reps=8):
    maes, w3, w4, meds = [], [], [], []
    for seed in range(reps):
        pr = np.full(len(X), np.nan, np.float32)
        ph = np.full(len(X), np.nan, np.float32)
        pe = np.full(len(X), np.nan, np.float32)
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
            pr[va] = fitg(ridge(), X[tr], y[tr], g[tr], X[va], g[va])
            ph[va] = fitg(hgbr(), X[tr], y[tr], g[tr], X[va], g[va])
            pe[va] = fitg(etr(), X[tr], y[tr], g[tr], X[va], g[va])
        blend = 0.5 * pr + 0.25 * ph + 0.25 * pe
        ae = np.abs(blend - y)
        maes.append(ae.mean()); w3.append((ae <= 3).mean()); w4.append((ae <= 4).mean()); meds.append(np.median(ae))
    return np.mean(maes), np.std(maes), np.mean(w3), np.mean(w4), np.mean(meds)


def main():
    E, Y, G, ID = load_emb()
    print(f"[wavlm2-train] speakers={len(Y)} emb_dim={E.shape[1]}", flush=True)
    m, sd, w3, w4, med = cv_blend(E, Y, G)
    print(f"  EMB-only       MAE={m:.3f}+/-{sd:.3f}  med={med:.2f}  w3={w3*100:.0f}% w4={w4*100:.0f}%", flush=True)
    res = {"emb_only": {"mae": m, "std": sd, "within3": w3, "within4": w4, "median": med}}

    phys = load_physics()
    P = np.stack([phys.get(i, np.zeros(9, np.float32)) for i in ID], 0)
    Pz = np.nan_to_num((P - np.nanmean(P, 0)) / (np.nanstd(P, 0) + 1e-6))
    Xf = np.concatenate([E, Pz], 1)
    m2, sd2, w32, w42, med2 = cv_blend(Xf, Y, G)
    print(f"  EMB+physics    MAE={m2:.3f}+/-{sd2:.3f}  med={med2:.2f}  w3={w32*100:.0f}% w4={w42*100:.0f}%", flush=True)
    res["emb_physics"] = {"mae": m2, "std": sd2, "within3": w32, "within4": w42, "median": med2}
    json.dump(res, open("outputs/wavlm2_result.json", "w"), indent=2)
    print("wrote outputs/wavlm2_result.json", flush=True)


if __name__ == "__main__":
    main()
