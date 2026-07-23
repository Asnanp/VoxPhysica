"""Test: does true AGE + GENDER + VOICE beat voice alone for height?
Loads features_audited (original clips), aggregates per speaker, joins age from npz.
Compares gender-split ridge variants. Writes outputs/age_voice.json.
"""
from __future__ import annotations
import os, glob, json, time
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

FEAT = "data/features_audited"
SCALARS = ["f0_mean", "formant_spacing_mean", "vtl_mean", "jitter", "shimmer",
           "hnr", "voiced_ratio", "duration_s", "snr_db_estimate"]


def fnum(z, k):
    try:
        return float(np.asarray(z[k]).item())
    except Exception:
        return float("nan")


def load(split):
    files = sorted(glob.glob(os.path.join(FEAT, split, "*.npz")))
    spk = {}
    t0 = time.time()
    for i, p in enumerate(files):
        with np.load(p, allow_pickle=True) as z:
            if "is_augmented" in z and int(np.asarray(z["is_augmented"]).item()) == 1:
                continue
            sid = str(np.asarray(z["speaker_id"]).item()).strip()
            h = fnum(z, "height_cm")
            if not sid or not np.isfinite(h):
                continue
            feat = [fnum(z, k) for k in SCALARS]
            if "sequence" in z:
                seq = np.asarray(z["sequence"], dtype=np.float32)
                with np.errstate(all="ignore"):
                    feat += list(np.nanmean(seq, axis=0)) + list(np.nanstd(seq, axis=0))
            e = spk.setdefault(sid, {"h": h, "g": fnum(z, "gender"), "age": fnum(z, "age"), "rows": []})
            e["rows"].append(np.asarray(feat, dtype=np.float32))
        if i % 6000 == 0:
            print(f"  [{split}] {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    X, y, g, age = [], [], [], []
    for sid, e in spk.items():
        X.append(np.nanmean(np.stack(e["rows"], 0), axis=0))
        y.append(e["h"]); g.append(e["g"]); age.append(e["age"])
    return (np.stack(X, 0).astype(np.float32), np.asarray(y, np.float32),
            np.asarray(g, np.float32), np.asarray(age, np.float32))


def mk():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", RobustScaler(quantile_range=(10, 90))),
                     ("m", RidgeCV(alphas=np.logspace(-2, 4, 25)))])


def gsplit(Xtr, ytr, gtr, Xte, gte):
    pred = np.full(len(Xte), np.nan, np.float32)
    for gv in np.unique(gtr):
        if not np.isfinite(gv):
            continue
        m = gtr == gv
        if m.sum() < 20:
            continue
        mdl = mk().fit(Xtr[m], ytr[m])
        mt = gte == gv
        if mt.any():
            pred[mt] = mdl.predict(Xte[mt])
    return np.where(np.isfinite(pred), pred, float(np.mean(ytr)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    Xtr, ytr, gtr, atr = load("train")
    Xva, yva, gva, ava = load("val")
    Xte, yte, gte, ate = load("test")
    print(f"speakers tr={len(ytr)} va={len(yva)} te={len(yte)} voice_dim={Xtr.shape[1]}", flush=True)

    def age_feats(a):
        a2 = np.nan_to_num(a, nan=float(np.nanmedian(atr)))
        return np.column_stack([a2, a2 ** 2, np.clip(a2, None, 25)]).astype(np.float32)  # age, age^2, growth-clip

    variants = {
        "A_voice_only": (Xtr, Xva, Xte),
        "B_voice+age": (np.column_stack([Xtr, age_feats(atr)]),
                         np.column_stack([Xva, age_feats(ava)]),
                         np.column_stack([Xte, age_feats(ate)])),
        "C_age+gender_only": (age_feats(atr), age_feats(ava), age_feats(ate)),
    }
    res = {}
    for name, (Tr, Va, Te) in variants.items():
        pv = gsplit(Tr, ytr, gtr, Va, gva)
        pt = gsplit(Tr, ytr, gtr, Te, gte)
        res[name] = {"val": mae(yva, pv), "test": mae(yte, pt),
                     "test_r2": float(r2_score(yte, pt)),
                     "within3cm": float(np.mean(np.abs(yte - pt) <= 3.0))}
        print(f"[{name:20s}] val={res[name]['val']:.3f} test={res[name]['test']:.3f} "
              f"R2={res[name]['test_r2']:+.3f} w3={res[name]['within3cm']*100:.0f}%", flush=True)

    delta = res["A_voice_only"]["test"] - res["B_voice+age"]["test"]
    print(f"\n>>> age contribution: {delta:+.3f} cm (voice_only - voice+age)", flush=True)
    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/age_voice.json", "w"), indent=2)


if __name__ == "__main__":
    main()
