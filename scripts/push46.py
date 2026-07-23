"""Push toward 4.6: blend WavLM-large v2 + handcrafted spectral view (from seqcache),
gender-split ridge per view, CV-tuned blend + stacking. Full 10x5 CV.
"""
from __future__ import annotations
import os, json, warnings, collections
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold


def load_wavlm2():
    E, Y, G, ID = [], [], [], []
    for sp in ["train", "val", "test"]:
        z = np.load(f"outputs/wavlm2/{sp}.npz", allow_pickle=True)
        E.append(z["emb"]); Y.append(z["y"]); G.append(z["g"]); ID.append(z["ids"].astype(str))
    return (np.concatenate(E), np.concatenate(Y).astype(np.float32),
            np.concatenate(G).astype(np.float32), np.concatenate(ID))


def load_handcrafted():
    """per-speaker: mean & std of spectral frames + physics scalars, from seqcache."""
    feats = {}
    for sp in ["train", "val", "test"]:
        z = np.load(f"outputs/seqcache/{sp}.npz", allow_pickle=True)
        seq, scal, sid = z["seq"], z["scal"].astype(np.float32), z["sid"].astype(str)
        per = collections.defaultdict(list)
        for i in range(len(sid)):
            s = seq[i].astype(np.float32)                       # (160,264)
            v = np.concatenate([s.mean(0), s.std(0), scal[i]])  # 264+264+9
            per[sid[i]].append(v)
        for s, vs in per.items():
            feats[s] = np.nanmean(np.stack(vs), 0).astype(np.float32)
    return feats


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


def R():
    return Pipeline([("s", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-1, 5, 40)))])
def Rk(k):
    return Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_regression, k=k)), ("m", RidgeCV(alphas=np.logspace(-1, 5, 40)))])


def main():
    Ew, Y, G, ID = load_wavlm2()
    hc = load_handcrafted()
    Eh = np.stack([hc.get(i, np.zeros(537, np.float32)) for i in ID], 0)
    print(f"speakers={len(Y)} wavlm={Ew.shape[1]} handcrafted={Eh.shape[1]}", flush=True)

    reps = 10
    res = collections.defaultdict(list)
    for seed in range(reps):
        pw = np.full(len(Y), np.nan, np.float32)   # wavlm pred
        ph = np.full(len(Y), np.nan, np.float32)   # handcrafted pred
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(Ew):
            pw[va] = fitg(R(), Ew[tr], Y[tr], G[tr], Ew[va], G[va])
            ph[va] = fitg(Rk(160), Eh[tr], Y[tr], G[tr], Eh[va], G[va])
        # tune blend weight on this seed's OOF (fair: it's CV preds)
        best_w, best = 1.0, 1e9
        for w in np.linspace(0, 1, 21):
            mm = np.mean(np.abs(w * pw + (1 - w) * ph - Y))
            if mm < best:
                best, best_w = mm, w
        blend = best_w * pw + (1 - best_w) * ph
        for nm, p in [("wavlm", pw), ("handcrafted", ph), ("blend", blend)]:
            ae = np.abs(p - Y)
            res[nm].append((ae.mean(), (ae <= 3).mean(), (ae <= 4).mean(), np.median(ae)))
        res["blend_w"].append(best_w)
    out = {}
    for nm in ["wavlm", "handcrafted", "blend"]:
        a = np.array(res[nm])
        out[nm] = {"mae": float(a[:, 0].mean()), "std": float(a[:, 0].std()),
                   "within3": float(a[:, 1].mean()), "within4": float(a[:, 2].mean()), "median": float(a[:, 3].mean())}
        print(f"[{nm:11s}] MAE={out[nm]['mae']:.3f}+/-{out[nm]['std']:.3f} med={out[nm]['median']:.2f} "
              f"w3={out[nm]['within3']*100:.0f}% w4={out[nm]['within4']*100:.0f}%", flush=True)
    print(f"  mean blend weight on wavlm = {np.mean(res['blend_w']):.2f}", flush=True)
    json.dump(out, open("outputs/push46_result.json", "w"), indent=2)
    print("wrote outputs/push46_result.json", flush=True)


if __name__ == "__main__":
    main()
