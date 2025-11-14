#!/usr/bin/env python3
"""
target_extractor_sep.py
Lightweight spectral-mask target speaker extractor.

Usage:
    python target_extractor_sep.py --mixture mixture.wav --target target.wav --outdir output_sep

Notes:
 - Requires your existing virtualenv with speechbrain, librosa, soundfile, numpy installed.
 - Tunable parameters: win_sec (window size for embeddings), hop_sec (hop for embeddings),
   mask_smooth (median filter on mask), sim_scale (sharpen similarity -> mask mapping).
"""

import os
import argparse
import tempfile
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import median_filter
from speechbrain.inference import EncoderClassifier

# ---------------------------
# Helpers
# ---------------------------

def load_audio_mono(path, sr=16000):
    audio, sr2 = librosa.load(path, sr=sr, mono=True)
    return audio, sr2

def write_wav(path, audio, sr=16000):
    sf.write(path, audio.astype(np.float32), sr)

def flatten_emb(x):
    a = np.asarray(x)
    if a.ndim > 1:
        return a.reshape(-1)
    return a

def cosine_similarity(a, b):
    a = flatten_emb(a)
    b = flatten_emb(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------------------------
# Main separator
# ---------------------------

def extract_with_spectral_mask(
    mixture_path,
    target_path,
    outdir,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    win_sec=0.4,
    hop_sec=0.2,
    sim_scale=6.0,
    mask_smooth=(1,9),
    sim_threshold=None
):
    os.makedirs(outdir, exist_ok=True)

    print("Loading models...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_ecapa"
    )

    print("Loading audio...")
    mix, _ = load_audio_mono(mixture_path, sr=sr)
    tgt, _ = load_audio_mono(target_path, sr=sr)

    # compute target embedding (use VAD-trimmed small region if needed)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write_wav(tmp.name, tgt, sr)
        targ_emb = model.encode_batch(model.load_audio(tmp.name))[0].detach().cpu().numpy().reshape(-1)
    try:
        os.unlink(tmp.name)
    except:
        pass

    # STFT of mixture
    print("Computing STFT...")
    S = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length, center=True)
    mag = np.abs(S)      # shape (freq_bins, time_frames)
    phase = np.angle(S)

    n_frames = mag.shape[1]
    frame_dur = hop_length / sr  # seconds per STFT frame
    total_dur = len(mix) / sr

    # Prepare embedding windows (time windows)
    win_len = int(round(win_sec * sr))
    hop_len = int(round(hop_sec * sr))
    starts = list(range(0, max(1, len(mix) - 1), hop_len))
    # keep windows that have at least 0.1s of audio
    windows = []
    for s in starts:
        e = min(s + win_len, len(mix))
        if e - s < int(0.06 * sr):  # skip too-short windows
            continue
        windows.append((s, e))

    print(f"Embedding windows: {len(windows)} (win={win_sec}s hop={hop_sec}s)")

    sims = []
    times = []  # center time (s) of each window
    tmp_files = []
    for (s, e) in windows:
        chunk = mix[s:e]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            write_wav(tmpf.name, chunk, sr)
            tmp_files.append(tmpf.name)
            emb = model.encode_batch(model.load_audio(tmpf.name))[0].detach().cpu().numpy().reshape(-1)
        sim = cosine_similarity(targ_emb, emb)
        center_t = (s + e) / 2.0 / sr
        sims.append(sim)
        times.append(center_t)

    # cleanup temp files
    for f in tmp_files:
        try:
            os.unlink(f)
        except:
            pass

    if len(sims) == 0:
        raise RuntimeError("No embedding windows were created (audio too short?)")

    sims = np.array(sims)
    # smooth sims (median filter)
    sims_sm = median_filter(sims, size=3)

    # derive threshold if not provided
    if sim_threshold is None:
        mean = sims_sm.mean()
        std = sims_sm.std()
        sim_threshold = max(0.35, mean + 0.12 * std)

    print(f"Similarity stats: mean={sims_sm.mean():.3f} std={sims_sm.std():.3f} thr={sim_threshold:.3f}")

    # Build time-aligned mask per STFT frame:
    # Map each STFT time frame center to nearest window index
    frame_times = np.arange(n_frames) * frame_dur + (frame_dur / 2.0)
    mask_vals = np.zeros(n_frames, dtype=float)

    # For each STFT frame, find nearest window center and take its similarity
    win_centers = np.array(times)
    for i, t in enumerate(frame_times):
        # find nearest window index
        idx = np.argmin(np.abs(win_centers - t))
        mask_vals[i] = sims_sm[idx]

    # sharpen mapping from similarity->mask using sigmoid
    def sim_to_mask(x, scale=sim_scale, thr=sim_threshold):
        # shift by threshold, then sigmoid-like
        z = (x - thr) * scale
        return 1.0 / (1.0 + np.exp(-z))
    mask_time = sim_to_mask(mask_vals)

    # Expand mask to freq bins
    mask_tf = np.tile(mask_time[np.newaxis, :], (mag.shape[0], 1))

    # Smooth mask in time-frequency domain (median filter)
    mask_tf = median_filter(mask_tf, size=mask_smooth)

    # Apply mask
    print("Applying mask and reconstructing audio...")
    target_mag = mag * mask_tf

    # reconstruct with original phase
    S_target = target_mag * np.exp(1j * phase)
    y_target = librosa.istft(S_target, hop_length=hop_length, length=len(mix))

    # post-processing: normalize
    if np.max(np.abs(y_target)) > 0:
        y_target = y_target / np.max(np.abs(y_target)) * 0.98

    # save outputs
    target_out = os.path.join(outdir, "target_speaker.wav")
    write_wav(target_out, y_target, sr)
    print("Wrote:", target_out)

    # write diarization-style json (per-window)
    diar = []
    for (s, e), sim, t in zip(windows, sims_sm, times):
        diar.append({"start": round(s / sr, 3), "end": round(e / sr, 3), "similarity": float(round(sim, 4)),
                     "mask_score": float(round(sim_to_mask(sim), 4))})

    with open(os.path.join(outdir, "diarization.json"), "w") as f:
        json.dump(diar, f, indent=2)

    return {"wav": target_out, "json": os.path.join(outdir, "diarization.json")}


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--outdir", default="output_sep")
    parser.add_argument("--win-sec", type=float, default=0.4, help="window size in seconds for embedding")
    parser.add_argument("--hop-sec", type=float, default=0.2, help="hop in seconds between embedding windows")
    parser.add_argument("--sim-scale", type=float, default=6.0, help="sharpening factor for sigmoid mapping")
    parser.add_argument("--mask-smooth-time", type=int, default=9, help="smoothing kernel time size (frames)")
    args = parser.parse_args()

    mask_smooth = (1, max(1, args.mask_smooth_time))
    out = extract_with_spectral_mask(
        args.mixture,
        args.target,
        args.outdir,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        sim_scale=args.sim_scale,
        mask_smooth=mask_smooth
    )
    print("Done. Outputs:", out)
