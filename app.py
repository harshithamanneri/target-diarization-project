# app.py
import os
import uuid
from flask import Flask, render_template, request, send_from_directory, abort
from target_extractor_v2 import extract_target_v2 as extract_target

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html", json_file=None, audio_file=None)


@app.route("/process", methods=["POST"])
def process():
    if "target" not in request.files or "mixture" not in request.files:
        return abort(400, "Missing 'target' or 'mixture' file")

    target_file = request.files["target"]
    mix_file = request.files["mixture"]

    if target_file.filename == "" or mix_file.filename == "":
        return abort(400, "Empty filename provided")

    job_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # save uploads
    target_path = os.path.join(UPLOAD_DIR, f"{job_id}_target.wav")
    mix_path = os.path.join(UPLOAD_DIR, f"{job_id}_mixture.wav")
    target_file.save(target_path)
    mix_file.save(mix_path)

    try:
        json_out, audio_out = extract_target(mix_path, target_path, job_dir, asr_model_name="tiny")
    except Exception as e:
        return abort(500, f"Processing failed: {e}")

    # relative URLs for frontend
    json_file = f"{job_id}/{os.path.basename(json_out)}"
    audio_file = f"{job_id}/{os.path.basename(audio_out)}"

    return render_template("index.html", json_file=json_file, audio_file=audio_file, job_id=job_id)


@app.route("/output/<jobid>/<filename>")
def serve_output(jobid, filename):
    jobpath = os.path.join(OUTPUT_DIR, jobid)
    if not os.path.exists(os.path.join(jobpath, filename)):
        return abort(404)
    return send_from_directory(jobpath, filename)


@app.route("/download/<jobid>/<filename>")
def download_output(jobid, filename):
    jobpath = os.path.join(OUTPUT_DIR, jobid)
    if not os.path.exists(os.path.join(jobpath, filename)):
        return abort(404)
    return send_from_directory(jobpath, filename, as_attachment=True)


if __name__ == "__main__":
    # run on a free port starting 5000
    port = 5000
    while True:
        try:
            app.run(debug=True, port=port)
            break
        except OSError:
            port += 1
