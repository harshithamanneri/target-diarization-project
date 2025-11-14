# socket_server.py (skeleton)
from flask import Flask
from flask_socketio import SocketIO, emit
import base64, numpy as np
from collections import defaultdict

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

buffers = defaultdict(bytes)  # per session buffer

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    sid = request.sid
    # data: {'pcm64': base64_string, 'sr':16000}
    pcm_bytes = base64.b64decode(data['pcm64'])
    buffers[sid] += pcm_bytes
    # when buffer is long enough, process VAD/embedding/ASR and emit results:
    # socketio.emit("transcript", {"text": "...", "speaker":"Target"}, room=sid)

@socketio.on("disconnect")
def on_disconnect():
    # cleanup
    buffers.pop(request.sid, None)
