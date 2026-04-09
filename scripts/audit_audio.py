import csv
import os
import glob
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Please run: pip install librosa soundfile numpy")
    exit()

def audit_audio():
    print("=== Automated Audio Quality Audit ===")
    
    with open('data/cleaned_dataset.csv', 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    print(f"Total Speakers: {len(rows)}")
    
    sample_rates = set()
    durations = []
    peak_amps = []
    channels = set()
    
    missing_files = 0
    total_files = 0
    
    # Audit a random subset to save time (max 1000 files)
    import random
    random.seed(42)
    random.shuffle(rows)
    
    for row in rows[:200]: # Check 200 speakers
        paths = row['audio_paths'].split('|')
        for p in paths:
            if not p.strip():
                continue
            total_files += 1
            if not os.path.exists(p):
                missing_files += 1
                continue
                
            try:
                # Load with soundfile to get exact raw specs without resampling
                info = sf.info(p)
                sample_rates.add(info.samplerate)
                channels.add(info.channels)
                durations.append(info.duration)
                
                # Load a tiny snippet to check amplitude max
                y, sr = sf.read(p, frames=10000)
                if len(y.shape) > 1:
                    y = y[:, 0]
                if len(y) > 0:
                    peak_amps.append(np.max(np.abs(y)))
                    
            except Exception as e:
                print(f"[ERROR] Could not read {p}: {e}")
                
    print(f"\n[RESULTS FROM {total_files} AUDITED FILES]")
    print(f"Missing Files   : {missing_files} / {total_files}")
    if sample_rates:
        print(f"Sample Rates    : {sample_rates}Hz")
    if channels:
        print(f"Audio Channels  : {channels}")
    if durations:
        print(f"Audio Durations : min={min(durations):.2f}s, max={max(durations):.2f}s, mean={np.mean(durations):.2f}s")
    if peak_amps:
        print(f"Peak Amplitudes : min={min(peak_amps):.4f}, max={max(peak_amps):.4f}, mean={np.mean(peak_amps):.4f}")
        
    print("\n[RECOMMENDATIONS FOR 10/10 QUALITY]")
    if len(sample_rates) > 1 or (16000 not in sample_rates and len(sample_rates) == 1):
        print("- [CRITICAL] You have mixed or non-16kHz sample rates! Resampling is MANDATORY.")
    else:
        print("- [OK] All audio is uniquely 16kHz.")
        
    if channels and max(channels) > 1:
        print("- [CRITICAL] Stereo audio detected! Must downmix to mono.")
    else:
        print("- [OK] All audio is mono.")
        
    if durations and min(durations) < 1.0:
        print(f"- [WARNING] Minimum duration is {min(durations):.2f}s. Very short clips carry no vocal tract info. We must drop/merge clips < 1.0s.")
        
    if peak_amps and (max(peak_amps) > 0.99 or min(peak_amps) < 0.1):
        print("- [CRITICAL] Peak amplitudes vary wildly (some very quiet, some clipping). RMS Volume Normalization is MANDATORY.")

if __name__ == '__main__':
    audit_audio()
