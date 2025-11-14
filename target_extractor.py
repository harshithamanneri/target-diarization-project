import os
import librosa
import numpy as np
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition

# Load ECAPA
recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed(audio, sr):
    """Convert waveform → ECAPA embedding"""
    tmp = "temp_embed.wav"
    sf.write(tmp, audio, sr)
    emb = recognizer.encode_batch(recognizer.load_audio(tmp))
    return emb.squeeze().detach().cpu().numpy()

def extract_target(mixture_path, target_path, outdir):
    """Main extraction (simple cosine similarity frame-based)."""

    # Load audio
    mix, sr1 = librosa.load(mixture_path, sr=16000, mono=True)
    tar, sr2 = librosa.load(target_path, sr=16000, mono=True)

    os.makedirs(outdir, exist_ok=True)

    # Compute target embedding
    tar_emb = embed(tar, 16000)

    # Split mixture into 1-sec chunks
    frame_len = 16000
    frames = []
    for i in range(0, len(mix), frame_len):
        start = i / 16000
        end = (i + frame_len) / 16000
        chunk = mix[i : i + frame_len]
        frames.append((start, end, chunk))

    diar = []
    collected = []

    for start, end, chunk in frames:
        if len(chunk) < 100:
            continue

        chunk_emb = embed(chunk, 16000)
        sim = cosine_similarity(tar_emb, chunk_emb)

        label = "Target" if sim > 0.55 else "Other"

        diar.append({
            "speaker": label,
            "start": round(start, 2),
            "end": round(end, 2),
            "similarity": round(sim, 3)
        })

        if label == "Target":
            collected.append(chunk)

    # Save diarization.json
    json_path = os.path.join(outdir, "diarization.json")

    import json
    with open(json_path, "w") as f:
        json.dump(diar, f, indent=4)

    # Save extracted voice
    if collected:
        out_audio = np.concatenate(collected)
    else:
        out_audio = np.zeros(16000)

    audio_path = os.path.join(outdir, "target_speaker.wav")
    sf.write(audio_path, out_audio, 16000)

    # RETURN correctly (no None!)
    return json_path, audio_path
