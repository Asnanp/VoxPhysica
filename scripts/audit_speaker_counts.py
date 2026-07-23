#!/usr/bin/env python
"""Count unique speakers per split and source."""
import numpy as np
import os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for split in ['train', 'val', 'test']:
    d = os.path.join(ROOT, 'data', 'features_v4', split)
    if not os.path.isdir(d):
        print(f"\n=== {split} === (directory not found)")
        continue

    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    speakers = set()
    sources = {}
    heights = []
    
    print(f"\n=== {split} ===")
    print(f"Total NPZ files: {len(files)}")

    for f in files:
        data = np.load(f, allow_pickle=True)
        sid = str(data['speaker_id'])
        h = float(data['height_cm'])
        src = str(data['source'])
        speakers.add(sid)
        sources.setdefault(src, set()).add(sid)
        heights.append(h)

    print(f"Total unique speakers: {len(speakers)}")
    for src, sps in sorted(sources.items()):
        print(f"  {src}: {len(sps)} unique speakers")
    if heights:
        print(f"Height range: {min(heights):.0f}-{max(heights):.0f} cm")
        print(f"Mean height: {np.mean(heights):.1f} cm")

# Check speaker overlap between splits
print("\n\n=== SPEAKER OVERLAP BETWEEN SPLITS ===")
split_speakers = {}
for split in ['train', 'val', 'test']:
    d = os.path.join(ROOT, 'data', 'features_v4', split)
    if not os.path.isdir(d):
        continue
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    sps = set()
    for f in files:
        data = np.load(f, allow_pickle=True)
        sps.add(str(data['speaker_id']))
    split_speakers[split] = sps
    print(f"{split}: {len(sps)} speakers")

# Cross-split overlap
for s1 in ['train', 'val', 'test']:
    for s2 in ['train', 'val', 'test']:
        if s1 >= s2:
            continue
        if s1 in split_speakers and s2 in split_speakers:
            overlap = split_speakers[s1] & split_speakers[s2]
            print(f"  {s1} ∩ {s2}: {len(overlap)} speakers (should be 0)")

# Count all labeled speakers across all splits
all_speakers = set()
for sps in split_speakers.values():
    all_speakers |= sps
print(f"\nTotal unique labeled speakers (all splits): {len(all_speakers)}")

# Count by source across all splits
all_sources = {}
for split in ['train', 'val', 'test']:
    d = os.path.join(ROOT, 'data', 'features_v4', split)
    if not os.path.isdir(d):
        continue
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    seen = set()
    for f in files:
        data = np.load(f, allow_pickle=True)
        sid = str(data['speaker_id'])
        if sid in seen:
            continue
        seen.add(sid)
        src = str(data['source'])
        all_sources.setdefault(src, set()).add(sid)

for src, sps in sorted(all_sources.items()):
    print(f"  Source '{src}': {len(sps)} total speakers (all splits)")
