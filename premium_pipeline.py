"""
premium_pipeline.py
Production-grade Target-Speaker Diarization & Extraction pipeline (Option C)

Usage:
    python premium_pipeline.py --mixture mixture.wav --target target.wav --outdir output_dir

Outputs:
    - output_dir/target_speaker.wav      # isolated target voice (best-separated source)
    - output_dir/diarization.json        # timeline with speaker labels, start/end, text, confidence
    - logs printed to console and saved (light)

Notes:
 - Requires Python 3.10-3.12 (3.12 recommended)
 - Recommended to run inside your .venv
"""

import os
import sys
import json
import tempfile
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

# Try to import heavy dependencies with graceful fallbacks
HAS_SPEECHBRAIN = True
HAS_PYANNO = True
HAS_WHISPER = True
HAS_TORCH = True

try:
    import torch
except Exception:
    HAS_TORCH = False

try:
    from speechbrain.pretrained import SepformerSeparation, SpeakerRecognition
except Exception:
    HAS_SPEECHBRAIN = False

try:
    # pyannote for high-quality diarization (may require HF token)
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    HAS_PYANNO = False

try:
    import whisper
except Exception:
    HAS_WHISPER = False

# ----------------------------
# Utility helpers
# ----------------------------
def load_audio_mono(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sr2 = librosa.load(path, sr=sr, mono=True)
    return audio, sr2

def write_wav(path: str, audio: np.ndarray, sr: int = 16000):
    sf.write(path, audio.astype(np.float32), sr)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_flatten_embedding(x):
    """Ensure numpy vector shape (N,)"""
    a = np.asarray(x)
    if a.ndim > 1:
        return a.flatten()
    return a

# ----------------------------
# Embedding & similarity
# ----------------------------
class EmbeddingExtractor:
    def __init__(self, savedir: str = "pretrained_models/spkrec_ecapa"):
        if not HAS_SPEECHBRAIN:
            raise RuntimeError("SpeechBrain not available. Install speechbrain.")
        # load once
        self.recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir
        )

    def file_embedding(self, path: str) -> np.ndarray:
        # SpeechBrain utilities accept files; use encode_file if available
        try:
            emb = self.recognizer.encode_file(path)
        except Exception:
            # fallback: write and use encode_batch
            emb = self.recognizer.encode_batch(self.recognizer.load_audio(path))
        emb = emb.squeeze().detach().cpu().numpy()
        return safe_flatten_embedding(emb)

    def array_embedding(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        # write temp file because SpeakerRecognition helpers often expect a file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            sf.write(t.name, waveform.astype(np.float32), sr)
            path = t.name
        try:
            emb = self.file_embedding(path)
        finally:
            try: os.unlink(path)
            except: pass
        return emb

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = safe_flatten_embedding(a)
    b = safe_flatten_embedding(b)
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

# ----------------------------
# Separation (SepFormer) - SpeechBrain
# ----------------------------
class Separator:
    def __init__(self, savedir="sepformer_model"):
        if not HAS_SPEECHBRAIN:
            raise RuntimeError("SpeechBrain not installed. pip install speechbrain")
        # Sepformer model (well-known speech separation)
        self.sep = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr",
            savedir=savedir
        )

    def separate(self, mix_path: str) -> List[np.ndarray]:
        """
        Returns list of separated waveforms (numpy arrays) at sample rate 8000/16000 depending on model.
        """
        est_sources = self.sep.separate_file(path=mix_path)  # shape (n_sources, T) or list
        # speechbrain returns torch tensor or numpy - normalize to numpy arrays
        if isinstance(est_sources, list):
            return [np.asarray(s) for s in est_sources]
        # if tensor: convert channels
        try:
            # shape (nsrc, time)
            arr = est_sources.detach().cpu().numpy()
            return [arr[i] for i in range(arr.shape[0])]
        except Exception:
            # final fallback: try to coerce
            return [np.asarray(est_sources)]

# ----------------------------
# Diarization: Pyannote (preferred) or simple VAD fallback
# ----------------------------
class Diarizer:
    def __init__(self, hf_token: Optional[str] = None):
        self.pipeline = None
        if HAS_PYANNO:
            try:
                if hf_token:
                    self.pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
                else:
                    # try without token (may work if model is public/cached)
                    self.pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization")
            except Exception as e:
                print("Pyannote pipeline failed to load:", e)
                self.pipeline = None

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Returns list of segments: [{"start": float, "end": float, "speaker": "SPEAKER_1"} ...]
        If pyannote available it yields higher-quality segmentation (with overlapping labels).
        Fallback: simple energy-based VAD to produce coarse segments.
        """
        if self.pipeline is not None:
            out = []
            diar = self.pipeline(audio_path)
            # diar is annotation with turns; iterate
            for turn, _, speaker in diar.itertracks(yield_label=True):
                out.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
            return out
        # fallback VAD segments (coarse)
        audio, sr = load_audio_mono(audio_path, sr=16000)
        frame_ms = 30
        frame_len = int(sr * frame_ms / 1000)
        energy = []
        for i in range(0, len(audio), frame_len):
            f = audio[i:i+frame_len]
            energy.append(float(np.mean(f**2)))
        thr = max(1e-6, np.median(energy) * 0.5)
        segments = []
        cur = None
        for i,e in enumerate(energy):
            t = i*frame_ms/1000.0
            if e>thr:
                if cur is None:
                    cur = [t, t + frame_ms/1000.0]
                else:
                    cur[1] = t + frame_ms/1000.0
            else:
                if cur is not None:
                    if cur[1]-cur[0] >= 0.12:
                        segments.append({"start": round(cur[0],3),"end": round(cur[1],3),"speaker":"UNK"})
                    cur = None
        if cur is not None and cur[1]-cur[0] >= 0.12:
            segments.append({"start": round(cur[0],3),"end": round(cur[1],3),"speaker":"UNK"})
        return segments

# ----------------------------
# ASR - Whisper
# ----------------------------
class ASR:
    def __init__(self, model_name="medium"):
        if not HAS_WHISPER:
            raise RuntimeError("Whisper not installed. pip install openai-whisper")
        self.model = whisper.load_model(model_name)

    def transcribe_file(self, wav_path: str) -> Tuple[str, float]:
        # returns (text, confidence)
        res = self.model.transcribe(wav_path, language=None, fp16=False)
        text = res.get("text", "").strip()
        # estimate confidence from segments if available; fallback 0.0..1.0
        conf = 0.0
        segments = res.get("segments", [])
        if segments:
            # use average of segment avg_logprob -> map to 0..1 via tanh-like mapping (approx)
            avg = sum((seg.get("avg_logprob", -5.0) for seg in segments))/len(segments)
            # convert avg_logprob (negative) to 0..1 range heuristically
            conf = float(1.0 / (1.0 + np.exp(- (avg + 5.0) )))  # rough mapping
        return text, conf

# ----------------------------
# Punctuation restoration (lightweight fallback)
# ----------------------------
def restore_punctuation_simple(text: str) -> str:
    # simple heuristic: capitalize after periods; ensure sentence ends with period.
    s = text.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in ".?!":
        s = s + "."
    return s

# ----------------------------
# Main pipeline
# ----------------------------
def run_premium_pipeline(
    mixture_path: str,
    target_path: str,
    outdir: str,
    hf_token: Optional[str] = None,
    sim_threshold: float = 0.6,
    whisper_model: str = "medium",
):
    ensure_dir(outdir)
    # 1) Load embeddings extractor
    print("[1/8] Loading speaker embedding model (ECAPA) ...")
    if not HAS_SPEECHBRAIN:
        raise RuntimeError("speechbrain unavailable. Install it first.")
    emb_extractor = EmbeddingExtractor()

    # target embedding
    print("[2/8] Computing target embedding ...")
    target_emb = emb_extractor.file_embedding(target_path)

    # 2) Run separation
    print("[3/8] Running source separation (SepFormer)...")
    separator = Separator()
    separated = separator.separate(mixture_path)  # list of numpy arrays
    print(f"    separated into {len(separated)} sources")

    # Save separated sources for inspection
    sep_paths = []
    for i, src in enumerate(separated):
        p = os.path.join(outdir, f"sep_source_{i+1}.wav")
        # If sampling rate differs, we assume 8000/16000; save at 16k
        write_wav(p, src, sr=16000)
        sep_paths.append(p)

    # 3) Compare embeddings between target and separated outputs
    print("[4/8] Matching separated sources to target embedding ...")
    best_i = 0
    best_sim = -1.0
    sims = []
    for i, p in enumerate(sep_paths):
        emb = emb_extractor.file_embedding(p)
        sim = cosine_similarity(emb, target_emb)
        sims.append(sim)
        print(f"    sim source {i+1} = {sim:.4f}")
        if sim > best_sim:
            best_sim = sim
            best_i = i

    print(f"    best match: source {best_i+1} with sim={best_sim:.4f}")

    # If best_sim below threshold, still proceed but warn
    if best_sim < sim_threshold:
        print(f"WARNING: best similarity {best_sim:.3f} < threshold {sim_threshold}. "
              "Output may contain other speakers or be low-quality.")

    target_source_path = os.path.join(outdir, "target_speaker.wav")
    # write chosen source as final target (first pass)
    os.replace(sep_paths[best_i], target_source_path)
    print(f"[5/8] Wrote preliminary target_speaker.wav -> {target_source_path}")

    # 4) Diarization to produce timeline (pyannote if available)
    print("[6/8] Running diarization ...")
    diarizer = Diarizer(hf_token=hf_token)
    segments = diarizer.diarize(mixture_path)
    # segments is list of {start,end,speaker}
    print(f"    diarization produced {len(segments)} segments")

    # 5) Map segments to speakers: for each segment, extract audio and determine whether it belongs to target
    # Build final JSON entries
    print("[7/8] Classifying segments vs target & running ASR ...")
    results = []
    asr = ASR(model_name=whisper_model) if HAS_WHISPER else None

    # load mixture audio once for segment cropping
    mix_audio, sr = load_audio_mono(mixture_path, sr=16000)

    for seg in segments:
        s = seg["start"]
        e = seg["end"]
        si = int(max(0, round(s * sr)))
        ei = int(min(len(mix_audio), round(e * sr)))
        chunk = mix_audio[si:ei]
        if len(chunk) == 0:
            continue

        # compare embedding of chunk vs target_emb and also vs the chosen separated source embedding
        chunk_emb = emb_extractor.array_embedding(chunk, sr=sr)
        chosen_emb = emb_extractor.file_embedding(target_source_path)
        sim_to_target = cosine_similarity(chunk_emb, target_emb)
        sim_to_chosen = cosine_similarity(chunk_emb, chosen_emb)

        # decide speaker label: prefer "Target" if sim_to_chosen >= sim_to_target and above threshold (or sim_to_target above threshold)
        is_target = (sim_to_chosen >= sim_to_target and sim_to_chosen >= sim_threshold) or (sim_to_target >= sim_threshold)

        label = "Target" if is_target else seg.get("speaker", "Other")

        # Write temp chunk for ASR
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpw:
            sf.write(tmpw.name, chunk.astype(np.float32), sr)
            tmp_path = tmpw.name

        # ASR
        text = ""
        confidence = 0.0
        if asr:
            try:
                text, confidence = asr.transcribe_file(tmp_path)
                text = restore_punctuation_simple(text)
            except Exception as e:
                print("ASR failed for segment:", e)
        else:
            text = ""
            confidence = 0.0

        # remove tmp file
        try:
            os.unlink(tmp_path)
        except:
            pass

        results.append({
            "speaker": label,
            "start": float(round(s, 3)),
            "end": float(round(e, 3)),
            "text": text,
            "confidence": float(round(confidence, 3))
        })

    # 6) Optionally post-process the extracted target audio: simple VAD to remove non-speech and concatenate target-labeled segments from separated source
    print("[8/8] Post-processing target audio (VAD cleanup & final write)...")
    # Extract target segments from the chosen separated source by matching with diarization segments
    chosen_src, sr_chosen = load_audio_mono(target_source_path, sr=16000)
    final_parts = []
    for seg in results:
        if seg["speaker"] == "Target":
            s = seg["start"]
            e = seg["end"]
            si = int(round(s * sr_chosen))
            ei = int(round(e * sr_chosen))
            si = max(0, si)
            ei = min(len(chosen_src), ei)
            if ei > si:
                final_parts.append(chosen_src[si:ei])
    if final_parts:
        final_audio = np.concatenate(final_parts)
        write_wav(target_source_path, final_audio, sr=16000)
        print("    final target_speaker.wav written (cleaned segments concatenated)")
    else:
        print("    WARNING: no target-labeled segments found; keeping preliminary separation output")

    # 7) Save JSON results
    json_path = os.path.join(outdir, "diarization.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Pipeline complete.")
    print("Outputs:")
    print(" -", target_source_path)
    print(" -", json_path)
    return {"target_wav": target_source_path, "diarization_json": json_path}

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture", required=True, help="Path to mixture wav")
    p.add_argument("--target", required=True, help="Path to short target ref wav (3-10s)")
    p.add_argument("--outdir", default="out", help="Output folder")
    p.add_argument("--hf-token", default=None, help="HuggingFace token (if required)")
    p.add_argument("--sim-thr", type=float, default=0.6, help="Similarity threshold (0..1)")
    p.add_argument("--whisper", default="medium", help="Whisper model size (tiny, base, small, medium, large)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_premium_pipeline(
            mixture_path=args.mixture,
            target_path=args.target,
            outdir=args.outdir,
            hf_token=args.hf_token,
            sim_threshold=args.sim_thr,
            whisper_model=args.whisper
        )
    except Exception as e:
        print("Pipeline failed:", e)
        traceback.print_exc()
        sys.exit(1)
