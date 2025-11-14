# main.py
import os
import argparse
from target_extractor_v2 import extract_target_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run target-diarization pipeline offline.")
    parser.add_argument("--mixture", required=True, help="Path to mixture.wav")
    parser.add_argument("--target", required=True, help="Path to target.wav (3-10s)")
    parser.add_argument("--outdir", default="output_final", help="Output directory")
    parser.add_argument("--asr", default="tiny", help="Whisper ASR model (tiny/base/small/medium/large)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    json_path, audio_path = extract_target_v2(args.mixture, args.target, args.outdir, asr_model_name=args.asr)

    print("\n=== FINISHED ===")
    print("Output files:")
    print(" - target_speaker.wav:", audio_path)
    print(" - diarization.json:", json_path)
