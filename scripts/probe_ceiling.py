"""Decisive ceiling probe: how much height signal exists in the features?

Loads speaker-level aggregated features, measures:
  1. within-gender Pearson r of physics features (vtl, formant_spacing, f0) vs height
  2. gender-mean baseline MAE (no-voice baseline)
  3. ridge / HGBR speaker-level MAE (gender-conditioned)
  4. linear ORACLE: ridge fit with CV (honest) + fit-on-test (cheating upper bound)
"""
from __future__ import annotations
import sys, os, math, json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.research.speaker_height_ensemble import load_speaker_split  # noqa

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

FEAT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/features_audited"


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def gender_of(meta):
    # gender stored as float scalar (1=male,0=female in NISP convention)
    return np.array([m.get("gender", float("nan")) for m in meta], dtype=float)


def main():
    print(f"[probe] loading {FEAT_DIR} ...", flush=True)
    tr = load_speaker_split(FEAT_DIR, "train")
    va = load_speaker_split(FEAT_DIR, "val", expected_feature_names=tr.feature_names)
    te = load_speaker_split(FEAT_DIR, "test", expected_feature_names=tr.feature_names)
    print(f"[probe] speakers train={len(tr.y)} val={len(va.y)} test={len(te.y)} feat_dim={tr.x.shape[1]}", flush=True)

    gtr, gva, gte = gender_of(tr.metadata), gender_of(va.metadata), gender_of(te.metadata)

    # ---- within-gender correlation of key physics features (using mean-aggregated cols) ----
    names = tr.feature_names
    def col(substr):
        idx = [i for i, n in enumerate(names) if substr in n and n.startswith("speaker_mean__")]
        return idx[0] if idx else None
    print("\n[ceiling] within-gender Pearson r (train speakers):")
    for feat in ["vtl_mean", "formant_spacing_mean", "f0_mean", "hnr", "jitter"]:
        c = col(feat)
        if c is None:
            print(f"   {feat:24s} (not found)"); continue
        line = f"   {feat:24s}"
        for gname, gval in [("male", 1), ("female", 0)]:
            m = gtr == gval
            if m.sum() < 5:
                line += f"  {gname}: n/a"; continue
            x = tr.x[m, c]; y = tr.y[m]
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else float("nan")
            line += f"  {gname}: r={r:+.3f} (n={int(m.sum())})"
        print(line)

    # ---- baselines ----
    gm = {g: tr.y[gtr == g].mean() for g in np.unique(gtr) if np.isfinite(g)}
    def gmean_pred(g): return np.array([gm.get(gi, tr.y.mean()) for gi in g])
    print("\n[baseline] predict per-gender train mean:")
    print(f"   val  MAE = {mae(va.y, gmean_pred(gva)):.3f} cm")
    print(f"   test MAE = {mae(te.y, gmean_pred(gte)):.3f} cm")

    # ---- gender-conditioned ridge + HGBR ----
    def make_ridge():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", RobustScaler(quantile_range=(10, 90))),
                         ("m", RidgeCV(alphas=np.logspace(-2, 4, 25)))])
    def make_hgbr():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", HistGradientBoostingRegressor(loss="absolute_error",
                              learning_rate=0.03, max_iter=500, max_leaf_nodes=15,
                              l2_regularization=0.1, random_state=0))])

    for label, mk in [("ridge", make_ridge), ("hgbr", make_hgbr)]:
        # pooled (single model, gender is a feature already)
        m = mk().fit(tr.x, tr.y)
        pv, pt = m.predict(va.x), m.predict(te.x)
        # gender-conditioned (separate model per gender)
        pv_g = np.zeros_like(va.y); pt_g = np.zeros_like(te.y)
        for g in np.unique(gtr):
            if not np.isfinite(g):
                continue
            mtr = gtr == g
            if mtr.sum() < 20:
                continue
            mg = mk().fit(tr.x[mtr], tr.y[mtr])
            mv, mt = gva == g, gte == g
            if mv.any(): pv_g[mv] = mg.predict(va.x[mv])
            if mt.any(): pt_g[mt] = mg.predict(te.x[mt])
        print(f"\n[{label}] pooled         val {mae(va.y, pv):.3f}  test {mae(te.y, pt):.3f}  (test R2={r2_score(te.y, pt):+.3f})")
        print(f"[{label}] gender-split   val {mae(va.y, pv_g):.3f}  test {mae(te.y, pt_g):.3f}")

    # ---- linear ORACLE: cheating upper bound (fit ridge ON test) ----
    pre = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler(quantile_range=(10, 90)))])
    Xte = pre.fit_transform(te.x)
    for a in [1.0, 10.0, 100.0]:
        orc = Ridge(alpha=a).fit(Xte, te.y)
        po = orc.predict(Xte)
        print(f"[ORACLE cheat fit-on-test ridge a={a:6.1f}] test MAE={mae(te.y, po):.3f}  R2={r2_score(te.y, po):+.3f}")
    print("\n(Oracle = absolute best a LINEAR model could do on these exact test features if it")
    print(" had peeked at the answers. Real models cannot beat this with linear signal.)")


if __name__ == "__main__":
    main()
