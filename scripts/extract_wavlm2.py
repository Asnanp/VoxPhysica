"""WavLM-large v2: more clips/speaker + layer-weighted (mean of mid layers).
Mid layers (speaker/physical traits) + more clips -> cleaner, richer speaker embedding.
Saves outputs/wavlm2/<split>.npz (emb 1024-d, y, g, src, ids).
"""
from __future__ import annotations
import os, time, argparse
import numpy as np, pandas as pd, torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--clips", type=int, default=16)
ap.add_argument("--maxsec", type=float, default=6.0)
ap.add_argument("--lo", type=int, default=6)    # layer band lo
ap.add_argument("--hi", type=int, default=18)   # layer band hi (inclusive)
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "outputs/wavlm2"; os.makedirs(OUT, exist_ok=True)
TS = 16000


def paths(r):
    return [p for p in str(r).split("|") if p.strip()]


def main():
    from transformers import AutoFeatureExtractor, WavLMModel
    print(f"[wavlm2] load on {DEV}, clips={args.clips}, layers[{args.lo}:{args.hi}]", flush=True)
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(DEV).eval()
    if DEV == "cuda":
        model = model.half()
    rng = np.random.default_rng(0)
    maxlen = int(args.maxsec * TS)
    for split in ["val", "test", "train"]:
        df = pd.read_csv(f"data/splits/{split}_clean.csv")
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
                if sr != TS:
                    wav = torchaudio.functional.resample(wav, sr, TS)
                x = wav[0].numpy()[:maxlen]
                if len(x) < TS // 2:
                    continue
                iv = fe(x, sampling_rate=TS, return_tensors="pt").input_values.to(DEV)
                if DEV == "cuda":
                    iv = iv.half()
                with torch.no_grad():
                    hs = model(iv).hidden_states           # tuple len 25, each (1,T,1024)
                band = torch.stack(hs[args.lo:args.hi + 1], 0).mean(0)   # (1,T,1024)
                vecs.append(band.float().mean(1)[0].cpu().numpy())
            if not vecs:
                continue
            embs.append(np.mean(np.stack(vecs, 0), 0))
            ys.append(float(row.height_cm))
            gs.append(1.0 if str(row.gender).lower().startswith("m") else 0.0)
            ss.append(str(row.source)); ids.append(str(row.speaker_id))
            if n % 30 == 0:
                print(f"  [{split}] {n}/{len(df)} {time.time()-t0:.0f}s", flush=True)
        E = np.stack(embs, 0).astype(np.float32)
        np.savez(f"{OUT}/{split}.npz", emb=E, y=np.asarray(ys, np.float32),
                 g=np.asarray(gs, np.float32), src=np.asarray(ss), ids=np.asarray(ids))
        print(f"[wavlm2] {split}: {E.shape} {time.time()-t0:.0f}s saved", flush=True)


if __name__ == "__main__":
    main()
