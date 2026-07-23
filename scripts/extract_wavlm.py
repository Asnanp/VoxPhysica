"""Extract WavLM-LARGE speaker embeddings (SOTA encoder) from raw audio.
Mean-pool last_hidden_state over time (1024-d) per clip, average over clips per speaker.
Saves outputs/wavlm/<split>.npz  (emb, y, g, src, ids).
Flags: --smoke (5 speakers/split), --clips N (max clips/speaker, default 12), --maxsec S (default 6)
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--clips", type=int, default=12)
ap.add_argument("--maxsec", type=float, default=6.0)
ap.add_argument("--model", default="microsoft/wavlm-large")
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "outputs/wavlm"
os.makedirs(OUT, exist_ok=True)


def paths(r):
    return [p for p in str(r).split("|") if p.strip()]


def main():
    from transformers import AutoFeatureExtractor, WavLMModel
    print(f"[wavlm] loading {args.model} on {DEV} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = WavLMModel.from_pretrained(args.model).to(DEV).eval()
    if DEV == "cuda":
        model = model.half()
    rng = np.random.default_rng(0)
    target_sr = 16000
    maxlen = int(args.maxsec * target_sr)

    for split in ["val", "test", "train"]:
        df = pd.read_csv(f"data/splits/{split}_clean.csv")
        if args.smoke:
            df = df.head(5)
        embs, ys, gs, ss, ids = [], [], [], [], []
        t0 = time.time()
        for n, row in enumerate(df.itertuples(index=False)):
            ps = paths(row.audio_paths)
            if len(ps) > args.clips:
                ps = list(rng.choice(ps, args.clips, replace=False))
            vecs = []
            for p in ps:
                if not os.path.exists(p):
                    continue
                try:
                    wav, sr = torchaudio.load(p)
                except Exception:
                    continue
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)
                x = wav[0].numpy()
                if len(x) > maxlen:
                    x = x[:maxlen]
                if len(x) < target_sr // 2:
                    continue
                iv = fe(x, sampling_rate=target_sr, return_tensors="pt").input_values.to(DEV)
                if DEV == "cuda":
                    iv = iv.half()
                with torch.no_grad():
                    h = model(iv).last_hidden_state  # (1,T,1024)
                vecs.append(h.float().mean(1)[0].cpu().numpy())
            if not vecs:
                continue
            embs.append(np.mean(np.stack(vecs, 0), 0))
            ys.append(float(row.height_cm))
            g = 1.0 if str(row.gender).lower().startswith("m") else 0.0
            gs.append(g); ss.append(str(row.source)); ids.append(str(row.speaker_id))
            if n % 20 == 0:
                print(f"  [{split}] {n}/{len(df)} {time.time()-t0:.0f}s", flush=True)
        E = np.stack(embs, 0).astype(np.float32)
        np.savez(f"{OUT}/{split}{'_smoke' if args.smoke else ''}.npz",
                 emb=E, y=np.asarray(ys, np.float32), g=np.asarray(gs, np.float32),
                 src=np.asarray(ss), ids=np.asarray(ids))
        print(f"[wavlm] {split}: {E.shape} speakers, {time.time()-t0:.0f}s -> saved", flush=True)


if __name__ == "__main__":
    main()
