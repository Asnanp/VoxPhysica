"""Lean ceiling probe. Fast: scalars + SSL + cheap seq mean/std, original clips only.
Writes results to outputs/ceiling_probe.json and prints a summary.
"""
from __future__ import annotations
import sys, os, glob, json, time
import numpy as np

FEAT = sys.argv[1] if len(sys.argv) > 1 else "data/features_audited"
INCLUDE_AUG = "--aug" in sys.argv
USE_SEQ = "--noseq" not in sys.argv
USE_SSL = "--nossl" not in sys.argv

SCALARS = ["f0_mean", "formant_spacing_mean", "vtl_mean", "jitter", "shimmer",
           "hnr", "voiced_ratio", "duration_s", "snr_db_estimate"]


def fnum(z, k):
    try:
        return float(np.asarray(z[k]).item())
    except Exception:
        return float("nan")


def load_split(split):
    files = sorted(glob.glob(os.path.join(FEAT, split, "*.npz")))
    spk = {}
    t0 = time.time()
    for i, p in enumerate(files):
        with np.load(p, allow_pickle=True) as z:
            if (not INCLUDE_AUG) and ("is_augmented" in z) and int(np.asarray(z["is_augmented"]).item()) == 1:
                continue
            sid = str(np.asarray(z["speaker_id"]).item()).strip()
            h = fnum(z, "height_cm")
            if not sid or not np.isfinite(h):
                continue
            feat = [fnum(z, k) for k in SCALARS]
            if USE_SEQ and "sequence" in z:
                seq = np.asarray(z["sequence"], dtype=np.float32)
                with np.errstate(all="ignore"):
                    feat += list(np.nanmean(seq, axis=0)) + list(np.nanstd(seq, axis=0))
            if USE_SSL and "ssl_embedding" in z:
                feat += list(np.asarray(z["ssl_embedding"], dtype=np.float32).ravel())
            g = fnum(z, "gender")
            src = str(np.asarray(z["source"]).item()).strip().upper() if "source" in z else "?"
            e = spk.setdefault(sid, {"h": h, "g": g, "src": src, "rows": []})
            e["rows"].append(np.asarray(feat, dtype=np.float32))
        if i % 4000 == 0:
            print(f"  [{split}] {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    X, y, gg, ids, srcs = [], [], [], [], []
    for sid, e in spk.items():
        R = np.stack(e["rows"], 0)
        X.append(np.nanmean(R, axis=0)); y.append(e["h"]); gg.append(e["g"]); ids.append(sid); srcs.append(e["src"])
    np.savez(os.path.join("outputs", f"_cache_{os.path.basename(FEAT)}_{split}.npz"),
             X=np.stack(X, 0).astype(np.float32), y=np.asarray(y, np.float32),
             g=np.asarray(gg, np.float32), src=np.asarray(srcs))
    return (np.stack(X, 0).astype(np.float32), np.asarray(y, np.float32),
            np.asarray(gg, np.float32), ids, np.asarray(srcs))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    from sklearn.linear_model import RidgeCV, Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score

    res = {"feat_dir": FEAT, "include_aug": INCLUDE_AUG, "use_seq": USE_SEQ, "use_ssl": USE_SSL}
    print(f"[lean] loading {FEAT} aug={INCLUDE_AUG} seq={USE_SEQ} ssl={USE_SSL}", flush=True)
    Xtr, ytr, gtr, _, str_ = load_split("train")
    Xva, yva, gva, _, sva = load_split("val")
    Xte, yte, gte, _, ste = load_split("test")
    print(f"[lean] speakers tr={len(ytr)} va={len(yva)} te={len(yte)} dim={Xtr.shape[1]}", flush=True)
    res["speakers"] = [len(ytr), len(yva), len(yte)]
    res["dim"] = int(Xtr.shape[1])

    # within-gender correlations (scalars are first len(SCALARS) columns)
    res["within_gender_r"] = {}
    print("\n[ceiling] within-gender Pearson r vs height (train):")
    for j, name in enumerate(SCALARS):
        row = {}
        line = f"   {name:22s}"
        for gn, gv in [("male", 1), ("female", 0)]:
            m = gtr == gv
            x, y = Xtr[m, j], ytr[m]
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")
            row[gn] = r; line += f"  {gn}:r={r:+.3f}"
        res["within_gender_r"][name] = row
        print(line, flush=True)

    # gender-mean baseline
    gm = {g: float(ytr[gtr == g].mean()) for g in np.unique(gtr) if np.isfinite(g)}
    def gpred(g): return np.array([gm.get(gi, ytr.mean()) for gi in g])
    res["baseline_gender_mean"] = {"val": mae(yva, gpred(gva)), "test": mae(yte, gpred(gte))}
    print(f"\n[baseline gender-mean] val={res['baseline_gender_mean']['val']:.3f} test={res['baseline_gender_mean']['test']:.3f}", flush=True)

    def ridge():
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", RobustScaler(quantile_range=(10, 90))),
                         ("m", RidgeCV(alphas=np.logspace(-2, 4, 25)))])
    def hgbr():
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("m", HistGradientBoostingRegressor(loss="absolute_error",
                              learning_rate=0.03, max_iter=400, max_leaf_nodes=15,
                              l2_regularization=0.1, random_state=0))])

    res["models"] = {}
    for label, mk in [("ridge", ridge), ("hgbr", hgbr)]:
        m = mk().fit(Xtr, ytr)
        pv, pt = m.predict(Xva), m.predict(Xte)
        pv_g, pt_g = np.zeros_like(yva), np.zeros_like(yte)
        for g in np.unique(gtr):
            if not np.isfinite(g):
                continue
            mtr = gtr == g
            if mtr.sum() < 20:
                continue
            mg = mk().fit(Xtr[mtr], ytr[mtr])
            mv, mt = gva == g, gte == g
            if mv.any(): pv_g[mv] = mg.predict(Xva[mv])
            if mt.any(): pt_g[mt] = mg.predict(Xte[mt])
        res["models"][label] = {
            "pooled_val": mae(yva, pv), "pooled_test": mae(yte, pt),
            "pooled_test_r2": float(r2_score(yte, pt)),
            "gsplit_val": mae(yva, pv_g), "gsplit_test": mae(yte, pt_g)}
        print(f"[{label}] pooled val={mae(yva,pv):.3f} test={mae(yte,pt):.3f} R2={r2_score(yte,pt):+.3f} | gsplit val={mae(yva,pv_g):.3f} test={mae(yte,pt_g):.3f}", flush=True)

    # source+gender conditioned ridge (exploit dataset structure)
    def ridge2():
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", RobustScaler(quantile_range=(10, 90))),
                         ("m", RidgeCV(alphas=np.logspace(-2, 4, 25)))])
    keytr = np.array([f"{s}|{int(g) if np.isfinite(g) else -1}" for s, g in zip(str_, gtr)])
    keyva = np.array([f"{s}|{int(g) if np.isfinite(g) else -1}" for s, g in zip(sva, gva)])
    keyte = np.array([f"{s}|{int(g) if np.isfinite(g) else -1}" for s, g in zip(ste, gte)])
    pv_sg, pt_sg = np.full_like(yva, np.nan), np.full_like(yte, np.nan)
    for k in np.unique(keytr):
        mtr = keytr == k
        if mtr.sum() < 20:
            continue
        mdl = ridge2().fit(Xtr[mtr], ytr[mtr])
        mv, mt = keyva == k, keyte == k
        if mv.any(): pv_sg[mv] = mdl.predict(Xva[mv])
        if mt.any(): pt_sg[mt] = mdl.predict(Xte[mt])
    # fallback for uncovered keys -> pooled gender-split ridge already computed mean
    fill = np.nanmean(ytr)
    pv_sg = np.where(np.isfinite(pv_sg), pv_sg, fill)
    pt_sg = np.where(np.isfinite(pt_sg), pt_sg, fill)
    w3v = float(np.mean(np.abs(pt_sg - yte) <= 3.0))
    res["src_gender_ridge"] = {"val": mae(yva, pv_sg), "test": mae(yte, pt_sg), "test_within3cm": w3v}
    print(f"\n[src+gender ridge] val={mae(yva,pv_sg):.3f} test={mae(yte,pt_sg):.3f} within3cm={w3v*100:.1f}%", flush=True)
    # report sources present
    print(f"   test sources: {dict(zip(*np.unique(ste, return_counts=True)))}", flush=True)

    # linear oracle (cheat: fit on test) -> absolute linear ceiling on these features
    pre = Pipeline([("i", SimpleImputer(strategy="median")), ("s", RobustScaler(quantile_range=(10, 90)))])
    Xt = pre.fit_transform(Xte)
    res["oracle"] = {}
    for a in [1.0, 10.0, 100.0]:
        o = Ridge(alpha=a).fit(Xt, yte)
        po = o.predict(Xt)
        res["oracle"][f"a{a}"] = {"mae": mae(yte, po), "r2": float(r2_score(yte, po))}
        print(f"[ORACLE fit-on-test a={a:5.1f}] test MAE={mae(yte,po):.3f} R2={r2_score(yte,po):+.3f}", flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ceiling_probe.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n[lean] wrote outputs/ceiling_probe.json", flush=True)


if __name__ == "__main__":
    main()
