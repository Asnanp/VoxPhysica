"""Cache per-clip frame sequences (T x 264) cropped to fixed L for our own NN.
Saves outputs/seqcache/<split>.npz : seq(N,L,264 fp16), scal(N,9), y, g, src, sid
"""
from __future__ import annotations
import os, glob, time
import numpy as np

FEAT = "data/features_vtl_ssl_augmented"
L = 160
SCAL = ["f0_mean", "formant_spacing_mean", "vtl_mean", "jitter", "shimmer", "hnr",
        "voiced_ratio", "duration_s", "snr_db_estimate"]
OUT = "outputs/seqcache"
os.makedirs(OUT, exist_ok=True)


def fnum(z, k):
    try:
        return float(np.asarray(z[k]).item())
    except Exception:
        return float("nan")


def crop(seq):
    seq = np.asarray(seq, np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    T = seq.shape[0]
    if T >= L:
        s = (T - L) // 2
        return seq[s:s + L]
    pad = np.zeros((L - T, seq.shape[1]), np.float32)
    return np.concatenate([seq, pad], 0)


def run(split):
    files = sorted(glob.glob(f"{FEAT}/{split}/*.npz"))
    seqs, scals, ys, gs, ss, sids = [], [], [], [], [], []
    t0 = time.time()
    for i, p in enumerate(files):
        with np.load(p, allow_pickle=True) as z:
            if "is_augmented" in z and int(np.asarray(z["is_augmented"]).item()) == 1:
                continue
            sid = str(np.asarray(z["speaker_id"]).item()).strip()
            h = fnum(z, "height_cm")
            if not sid or not np.isfinite(h) or "sequence" not in z:
                continue
            seqs.append(crop(z["sequence"]).astype(np.float16))
            scals.append(np.asarray([fnum(z, k) for k in SCAL], np.float32))
            ys.append(h); gs.append(fnum(z, "gender"))
            ss.append(str(np.asarray(z["source"]).item()) if "source" in z else "?")
            sids.append(sid)
        if i % 4000 == 0:
            print(f"  [{split}] {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{OUT}/{split}.npz",
             seq=np.stack(seqs, 0), scal=np.stack(scals, 0),
             y=np.asarray(ys, np.float32), g=np.asarray(gs, np.float32),
             src=np.asarray(ss), sid=np.asarray(sids))
    print(f"[seqcache] {split}: {len(ys)} clips -> saved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for s in ["val", "test", "train"]:
        run(s)
