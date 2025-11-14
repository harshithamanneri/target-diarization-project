import librosa
import soundfile as sf
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import os
import tempfile
import json

# Load ECAPA-TDNN speaker recognition model
recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_model"
)

def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio, sr

def get_embedding(audio):
    """
    Generates a flattened speaker embedding for compatibility
    with SpeechBrain v1.0 output shape (1,192).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        emb = recognizer.encode_batch(recognizer.load_audio(tmp.name))

    emb = emb.squeeze().detach().cpu().numpy()

    # Flatten: (1,192) -> (192,)
    emb = emb.flatten()
    return emb

def split_chunks(audio, sr, sec=1):
    L = int(sr * sec)
    chunks = []
    for i in range(0, len(audio), L):
        part = audio[i:i+L]
        if len(part) > 0:
            start = i / sr
            end = (i + len(part)) / sr
            chunks.append((start, end, part))
    return chunks

def cosine_similarity(a, b):
    """
    Safe cosine similarity: handles zero vectors, NaN, etc.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    sim = np.dot(a, b) / denom
    return float(sim)

def run_diarization(mix_path, target_path, output_folder):
    # Load audio
    mix, sr = load_audio(mix_path)
    target, _ = load_audio(target_path)

    # Compute embeddings
    target_emb = get_embedding(target)

    # Split mixture
    chunks = split_chunks(mix, sr)

    results = []
    extracted = []

    for start, end, chunk in chunks:
        emb = get_embedding(chunk)

        similarity = cosine_similarity(emb, target_emb)

        label = "Target" if similarity > 0.55 else "Other"

        results.append({
            "speaker": label,
            "start": round(start, 2),
            "end": round(end, 2),
            "similarity": round(similarity, 3)
        })

        if label == "Target":
            extracted.append(chunk)

    # Save diarization JSON
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "diarization.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save target-only audio
    if extracted:
        combined = np.concatenate(extracted)
        sf.write(os.path.join(output_folder, "target_speaker.wav"), combined, sr)
    else:
        sf.write(os.path.join(output_folder, "target_speaker.wav"), np.zeros(16000), sr)
