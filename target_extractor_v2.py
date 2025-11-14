# target_extractor_v2.py
import os
import json
import numpy as np
import soundfile as sf
import librosa
from speechbrain.inference import EncoderClassifier
import whisper
from scipy.signal import medfilt


# ---------------------------
# Helpers
# ---------------------------
SR = 16000


def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def simple_vad(audio, sr=SR, frame_ms=30, energy_thresh=3e-4):
    frame_len = int(sr * frame_ms / 1000)
    voiced = []
    for i in range(0, len(audio), frame_len):
        frame = audio[i:i + frame_len]
        if len(frame) == 0:
            continue
        energy = float(np.mean(frame**2))
        if energy > energy_thresh:
            voiced.append(frame)
    if len(voiced) == 0:
        return audio
    return np.concatenate(voiced)


def chunk_audio(audio, sr=SR, sec=1.0):
    L = int(sr * sec)
    chunks = []
    for i in range(0, len(audio), L):
        chunk = audio[i:i + L]
        if len(chunk) > 0:
            start = i / sr
            end = (i + len(chunk)) / sr
            chunks.append((i, start, end, chunk))
    return chunks


def cosine_similarity(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def smooth(values, win=3):
    if len(values) == 0:
        return values
    # pad and median
    padded = np.array(values)
    return medfilt(padded, kernel_size=min(max(1, win), len(values) if len(values)%2==1 else len(values)-1)).tolist()


# ---------------------------
# Main extraction function
# ---------------------------
def extract_target_v2(mixture_path, target_path, outdir,
                      asr_model_name="tiny", chunk_sec=1.0, vad_energy=3e-4,
                      similarity_floor=0.35, adapt_sigma_scale=0.15):
    """
    Runs target extraction:
    - loads ECAPA model to compute embeddings
    - splits mixture into fixed-length chunks
    - computes similarity to target embedding per chunk
    - smooths similarities and sets adaptive threshold
    - extracts audio for chunks labeled Target
    - runs Whisper ASR per chunk and records text+confidence
    Returns (json_path, audio_path)
    """

    os.makedirs(outdir, exist_ok=True)

    # 1) load models
    print("🔊 Loading ECAPA-TDNN (speaker embedding) ...")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(outdir, "pretrained_ecapa"),
        run_opts={"device":"cpu"},
    )

    print(f"🔤 Loading Whisper ASR model: {asr_model_name} ...")
    asr = whisper.load_model(asr_model_name)

    # 2) load audio
    print("🎧 Loading mixture & target ...")
    mix = load_audio(mixture_path)
    tar = load_audio(target_path)

    # 3) quick VAD on target to remove silence/clips
    print("🧹 Running VAD on target sample ...")
    tar_vad = simple_vad(tar, sr=SR, energy_thresh=vad_energy)
    tmp_target = os.path.join(outdir, "temp_target_vad.wav")
    sf.write(tmp_target, tar_vad, SR)

    # 4) compute target embedding
    print("🔎 Computing target embedding ...")
    tar_emb = ecapa.encode_batch(ecapa.load_audio(tmp_target))
    tar_emb = tar_emb[0].detach().cpu().numpy().reshape(-1)

    # 5) chunk mixture
    print(f"✂ Splitting mixture into {chunk_sec:.2f}s chunks ...")
    chunks = chunk_audio(mix, sr=SR, sec=chunk_sec)
    if len(chunks) == 0:
        raise RuntimeError("Mixture produced zero chunks (empty audio?)")

    similarities = []
    json_results = []
    extracted_chunks = []

    # 6) compute similarity per chunk
    print("🔬 Computing similarities per chunk ...")
    for idx, (sample_idx, start, end, chunk) in enumerate(chunks):
        temp_chunk_path = os.path.join(outdir, f"temp_chunk_{idx}.wav")
        sf.write(temp_chunk_path, chunk, SR)

        emb = ecapa.encode_batch(ecapa.load_audio(temp_chunk_path))
        emb = emb[0].detach().cpu().numpy().reshape(-1)
        sim = cosine_similarity(tar_emb, emb)
        similarities.append(sim)
        json_results.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "raw_similarity": float(sim)
        })
        print(f"  chunk {idx+1}/{len(chunks)}  start={start:.2f}s sim={sim:.4f}")

    # 7) smooth similarities and compute adaptive threshold
    print("\n📈 Smoothing similarities ...")
    smoothed = smooth(similarities, win=3)
    mean = float(np.mean(smoothed))
    sigma = float(np.std(smoothed))
    thr = max(similarity_floor, mean + adapt_sigma_scale * sigma)
    print(f"⚙️ Adaptive threshold = {thr:.4f} (mean={mean:.4f} std={sigma:.4f})")

    # 8) label chunks and collect target audio
    for i, entry in enumerate(json_results):
        sim_val = float(smoothed[i]) if i < len(smoothed) else float(entry.get("raw_similarity", 0.0))
        label = "Target" if sim_val >= thr else "Other"
        entry["similarity"] = round(sim_val, 4)
        entry["speaker"] = label

    for entry, (_, _, _, chunk) in zip(json_results, chunks):
        if entry["speaker"] == "Target":
            extracted_chunks.append(chunk)

    # 9) write extracted audio (concatenate)
    audio_out_path = os.path.join(outdir, "target_speaker.wav")
    if len(extracted_chunks) > 0:
        out_audio = np.concatenate(extracted_chunks)
        # normalize mildly
        peak = np.max(np.abs(out_audio)) if out_audio.size else 1.0
        if peak > 0:
            out_audio = out_audio / (peak + 1e-9) * 0.98
        sf.write(audio_out_path, out_audio, SR)
        print(f"\n✅ Wrote extracted audio -> {audio_out_path}")
    else:
        # no target detected: write 1s silence to maintain file
        sf.write(audio_out_path, np.zeros(SR), SR)
        print("\n⚠ No target segments detected. Wrote silent file.")

    # 10) Run ASR per chunk and populate text + confidence
    print("\n📝 Running ASR for each chunk (to fill text & confidence)...")
    for idx, entry in enumerate(json_results):
        s = entry["start"]
        e = entry["end"]
        i_sample = int(s * SR)
        j_sample = int(e * SR)
        audio_chunk = mix[i_sample:j_sample]
        temp_chunk_path = os.path.join(outdir, f"asr_chunk_{idx}.wav")
        sf.write(temp_chunk_path, audio_chunk, SR)

        try:
            res = asr.transcribe(temp_chunk_path)
            text = res.get("text", "").strip()
            # compute a simple confidence: use avg_logprob from first segment if present
            confidence = 0.0
            if res.get("segments"):
                segs = res["segments"]
                # convert avg_logprob (~negative) into pseudo confidence
                avg_logprob = np.mean([seg.get("avg_logprob", -1.0) for seg in segs])
                # map avg_logprob (typically -1..-10) to [0,1]
                confidence = float(np.clip(1.0 + avg_logprob / 5.0, 0.0, 1.0))
            else:
                # fallback confidence estimate using length
                confidence = float(min(0.9, max(0.2, len(text) / 100.0)))
        except Exception as ex:
            text = ""
            confidence = 0.0
            print(f"ASR failed on chunk {idx}: {ex}")

        entry["text"] = text
        entry["confidence"] = round(float(confidence), 3)

    # 11) Save JSON (conform to professor's required fields)
    json_out_path = os.path.join(outdir, "diarization.json")

    # Convert to desired schema: keep all entries with required keys
    final_entries = []
    for e in json_results:
        final_entries.append({
            "speaker": e.get("speaker", "Other"),
            "start": float(e.get("start", 0.0)),
            "end": float(e.get("end", 0.0)),
            "text": e.get("text", ""),
            "confidence": float(e.get("confidence", 0.0))
        })

    with open(json_out_path, "w") as jf:
        json.dump(final_entries, jf, indent=4)

    print(f"\n📄 Wrote diarization JSON -> {json_out_path}")

    return json_out_path, audio_out_path


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--outdir", default="output_final")
    parser.add_argument("--asr", default="tiny", help="whisper model: tiny, base, small, medium, large")
    args = parser.parse_args()

    extract_target_v2(args.mixture, args.target, args.outdir, asr_model_name=args.asr)
