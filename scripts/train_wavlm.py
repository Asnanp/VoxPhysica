"""Train height regressor on WavLM-large embeddings (SOTA encoder).
Reports: gender-split ridge (test + full-CV) and a deep MLP (test + full-CV).
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold

W = "outputs/wavlm"


def load(split):
    z = np.load(f"{W}/{split}.npz", allow_pickle=True)
    return z["emb"].astype(np.float32), z["y"].astype(np.float32), z["g"].astype(np.float32), z["ids"]


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def ridge():
    return Pipeline([("s", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-1, 5, 40)))])


def fit_gender(mk, Xtr, ytr, gtr, Xq, gq):
    p = np.full(len(Xq), np.nan, np.float32)
    for gv in np.unique(gtr):
        m = gtr == gv
        if m.sum() < 25:
            continue
        f = clone(mk).fit(Xtr[m], ytr[m]); mq = gq == gv
        if mq.any():
            p[mq] = f.predict(Xq[mq])
    return np.where(np.isfinite(p), p, float(np.mean(ytr)))


def full_cv(X, y, g, mk, reps=8):
    out = []
    for seed in range(reps):
        pred = np.full(len(X), np.nan, np.float32)
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
            pred[va] = fit_gender(mk, X[tr], y[tr], g[tr], X[va], g[va])
        out.append(np.abs(pred - y))
    A = np.stack(out, 0)
    return float(A.mean()), float(A.mean(1).std()), float((A <= 3).mean()), float((A <= 4).mean())


def mlp_cv(X, y, g):
    """Deep gender-conditioned MLP, full-CV via torch."""
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.g = nn.Embedding(2, 16)
            s.net = nn.Sequential(
                nn.Linear(d + 16, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, 1))

        def forward(s, x, gi):
            return s.net(torch.cat([x, s.g(gi)], 1)).squeeze(1)

    def run(Xtr, ytr, gtr, Xte, gte, ymean, ystd):
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        net = Net(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        Xt = torch.tensor(Xtr2, dtype=torch.float32, device=dev)
        yt = torch.tensor((ytr - ymean) / ystd, dtype=torch.float32, device=dev)
        gt = torch.tensor(gtr.astype(int), device=dev)
        lossf = nn.SmoothL1Loss()
        net.train()
        for ep in range(300):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                out = net(Xt[idx], gt[idx])
                loss = lossf(out, yt[idx])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor(Xte2, dtype=torch.float32, device=dev),
                     torch.tensor(gte.astype(int), device=dev)).cpu().numpy()
        return pr * ystd + ymean

    out = []
    for seed in range(4):
        pred = np.full(len(X), np.nan, np.float32)
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
            ym, ys = y[tr].mean(), y[tr].std()
            pred[va] = run(X[tr], y[tr], g[tr], X[va], g[va], ym, ys)
        out.append(np.abs(pred - y))
    A = np.stack(out, 0)
    return float(A.mean()), float(A.mean(1).std()), float((A <= 3).mean()), float((A <= 4).mean())


def main():
    Xtr, ytr, gtr, _ = load("train")
    Xva, yva, gva, _ = load("val")
    Xte, yte, gte, _ = load("test")
    print(f"[wavlm-train] tr={len(ytr)} va={len(yva)} te={len(yte)} dim={Xtr.shape[1]}", flush=True)
    Xp = np.concatenate([Xtr, Xva]); yp = np.concatenate([ytr, yva]); gp = np.concatenate([gtr, gva])
    Xall = np.concatenate([Xtr, Xva, Xte]); yall = np.concatenate([ytr, yva, yte]); gall = np.concatenate([gtr, gva, gte])

    rt = fit_gender(ridge(), Xp, yp, gp, Xte, gte)
    print(f"  RIDGE  fixed-test MAE={mae(yte,rt):.3f}", flush=True)
    m, sd, w3, w4 = full_cv(Xall, yall, gall, ridge())
    print(f"  RIDGE  full-CV MAE={m:.3f}+/-{sd:.3f}  w3={w3*100:.0f}% w4={w4*100:.0f}%", flush=True)
    res = {"ridge_test": mae(yte, rt), "ridge_cv": m, "ridge_cv_std": sd, "ridge_w3": w3, "ridge_w4": w4}

    try:
        mm, msd, mw3, mw4 = mlp_cv(Xall, yall, gall)
        print(f"  MLP    full-CV MAE={mm:.3f}+/-{msd:.3f}  w3={mw3*100:.0f}% w4={mw4*100:.0f}%", flush=True)
        res.update({"mlp_cv": mm, "mlp_cv_std": msd, "mlp_w3": mw3, "mlp_w4": mw4})
    except Exception as e:
        print("  MLP failed:", e, flush=True)
    json.dump(res, open("outputs/wavlm_result.json", "w"), indent=2)
    print("wrote outputs/wavlm_result.json", flush=True)


if __name__ == "__main__":
    main()
