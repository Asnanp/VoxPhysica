"""VoxHeightNet - our own from-scratch NN for height-from-voice.
Conv1d temporal encoder + self-attention pooling, fused with a physics scalar branch
and gender embedding, multi-task aux heads (gender/age/weight) for regularization.
Clip-level training, speaker-level evaluation. No pretrained weights.
"""
from __future__ import annotations
import os, json, math
import numpy as np
import torch
import torch.nn as nn

SC = "outputs/seqcache"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def load(split):
    z = np.load(f"{SC}/{split}.npz", allow_pickle=True)
    return (z["seq"], z["scal"].astype(np.float32), z["y"].astype(np.float32),
            z["g"].astype(np.float32), z["sid"].astype(str))


class AttnPool(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.q = nn.Linear(d, 1)

    def forward(s, x, mask=None):           # x: (B,T,D)
        w = s.q(x).squeeze(-1)              # (B,T)
        a = torch.softmax(w, dim=1).unsqueeze(-1)
        return (x * a).sum(1)               # (B,D)


class VoxHeightNet(nn.Module):
    def __init__(s, in_ch=264, n_scal=9, hid=192):
        super().__init__()
        def blk(ci, co, k=5, st=2):
            return nn.Sequential(nn.Conv1d(ci, co, k, st, k // 2), nn.BatchNorm1d(co), nn.GELU(), nn.Dropout(0.15))
        s.enc = nn.Sequential(blk(in_ch, 128), blk(128, 192), blk(192, hid), blk(hid, hid))
        s.attn = nn.MultiheadAttention(hid, 4, batch_first=True, dropout=0.1)
        s.norm = nn.LayerNorm(hid)
        s.pool = AttnPool(hid)
        s.phys = nn.Sequential(nn.Linear(n_scal, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU())
        s.gemb = nn.Embedding(2, 16)
        fd = hid + 48 + 16
        s.trunk = nn.Sequential(nn.Linear(fd, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
                                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3))
        s.head_h = nn.Linear(128, 1)
        s.head_g = nn.Linear(128, 1)
        s.head_a = nn.Linear(128, 1)   # age aux
        s.head_w = nn.Linear(128, 1)   # weight aux

    def forward(s, seq, scal, g):
        x = seq.transpose(1, 2)            # (B,264,T)
        h = s.enc(x).transpose(1, 2)       # (B,T',hid)
        att, _ = s.attn(h, h, h)
        h = s.norm(h + att)
        clip = s.pool(h)                   # (B,hid)
        p = s.phys(scal)
        z = torch.cat([clip, p, s.gemb(g)], 1)
        z = s.trunk(z)
        return s.head_h(z).squeeze(1), s.head_g(z).squeeze(1), s.head_a(z).squeeze(1), s.head_w(z).squeeze(1)


def speaker_mae(pred_clip, y_clip, sid):
    import collections
    agg = collections.defaultdict(list); ya = {}
    for p, y, s in zip(pred_clip, y_clip, sid):
        agg[s].append(p); ya[s] = y
    ae = [abs(np.mean(v) - ya[s]) for s, v in agg.items()]
    ae = np.array(ae)
    return float(ae.mean()), float(np.median(ae)), float((ae <= 3).mean()), float((ae <= 4).mean())


def main():
    seq_tr, sc_tr, y_tr, g_tr, sid_tr = load("train")
    seq_va, sc_va, y_va, g_va, sid_va = load("val")
    seq_te, sc_te, y_te, g_te, sid_te = load("test")
    print(f"[voxnet] clips tr={len(y_tr)} va={len(y_va)} te={len(y_te)} seq={seq_tr.shape[1:]}", flush=True)

    # standardize scalars + height
    sm, ss = np.nanmean(sc_tr, 0), np.nanstd(sc_tr, 0) + 1e-6
    sc_tr = np.nan_to_num((sc_tr - sm) / ss); sc_va = np.nan_to_num((sc_va - sm) / ss); sc_te = np.nan_to_num((sc_te - sm) / ss)
    ym, ysd = float(y_tr.mean()), float(y_tr.std())

    def T(a, dt=torch.float32):
        return torch.tensor(a, dtype=dt)
    Xs = T(seq_tr.astype(np.float32))            # keep on CPU, move per batch
    net = VoxHeightNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=2e-2)
    EP, BS = 60, 64
    steps = math.ceil(len(y_tr) / BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2.5e-3, epochs=EP, steps_per_epoch=steps)
    huber = nn.SmoothL1Loss(); bce = nn.BCEWithLogitsLoss()
    sc_tr_t, g_tr_t = T(sc_tr), T(g_tr.astype(np.int64), torch.long)
    yz = T((y_tr - ym) / ysd)

    def eval_split(seq, sc, y, g, sid):
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(y), 256):
                sq = T(seq[i:i + 256].astype(np.float32)).to(DEV)
                p, *_ = net(sq, T(sc[i:i + 256]).to(DEV), T(g[i:i + 256].astype(np.int64), torch.long).to(DEV))
                preds.append(p.cpu().numpy() * ysd + ym)
        return speaker_mae(np.concatenate(preds), y, sid)

    best = (1e9, None); best_ep = -1
    for ep in range(EP):
        net.train(); perm = np.random.permutation(len(y_tr))
        for i in range(0, len(y_tr), BS):
            idx = perm[i:i + BS]
            sq = Xs[idx].to(DEV)
            with torch.autocast(DEV, enabled=(DEV == "cuda")):
                ph, pg, pa, pw = net(sq, sc_tr_t[idx].to(DEV), g_tr_t[idx].to(DEV))
                loss = huber(ph, yz[idx].to(DEV)) + 0.15 * bce(pg, g_tr_t[idx].float().to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        vm, vmed, vw3, vw4 = eval_split(seq_va, sc_va, y_va, g_va, sid_va)
        if vm < best[0]:
            best = (vm, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}); best_ep = ep
        if ep % 5 == 0 or ep == EP - 1:
            print(f"  ep{ep:02d} val_spkMAE={vm:.3f} med={vmed:.2f} w3={vw3*100:.0f}% w4={vw4*100:.0f}%  (best {best[0]:.3f}@{best_ep})", flush=True)

    net.load_state_dict(best[1])
    vm = eval_split(seq_va, sc_va, y_va, g_va, sid_va)
    tm = eval_split(seq_te, sc_te, y_te, g_te, sid_te)
    print(f"\n[VoxHeightNet] BEST val speaker MAE={vm[0]:.3f}", flush=True)
    print(f"[VoxHeightNet] TEST speaker MAE={tm[0]:.3f} median={tm[1]:.2f} within3cm={tm[2]*100:.0f}% within4cm={tm[3]*100:.0f}%", flush=True)
    torch.save(best[1], "outputs/voxheightnet.pt")
    json.dump({"val_mae": vm[0], "test_mae": tm[0], "test_median": tm[1],
               "test_within3cm": tm[2], "test_within4cm": tm[3], "best_epoch": best_ep},
              open("outputs/voxnet_result.json", "w"), indent=2)
    print("wrote outputs/voxnet_result.json + voxheightnet.pt", flush=True)


if __name__ == "__main__":
    main()
