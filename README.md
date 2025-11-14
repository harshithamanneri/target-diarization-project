# 🎤 Target Speaker Diarization Web Application  
Extract the **target speaker’s voice** from a multi-speaker mixture using a clean and simple web interface.

This project allows a user to upload:

1. **Target Speaker Audio** – Audio of the speaker you want to isolate  
2. **Mixture Audio** – Audio containing multiple speakers  

The system then extracts the voice **only of the target speaker** and generates:

- ✔ `target_speaker.wav` — Extracted clean voice  
- ✔ `diarization.json` — Timestamps + similarity scores  
- ✔ Beautiful web UI for easy usage  

---

## 🚀 Demo Screenshot

> *(Add a screenshot here if needed.)*

---

## 🛠 Features

- 🎙 Upload two audio files (target + mixture)
- 🔍 Detect the target speaker inside mixture
- ✂ Extract only their speech segments
- 🧾 Generate diarization JSON  
- 🌐 Fully working Flask-based web application  
- 🎨 Clean, responsive UI (HTML + CSS)
- ⚡ Works completely offline  

---

## 📂 Project Structure

```
project/
│
├── app.py                 # Flask backend
├── diarization.py         # Logic for diarization + extraction
├── main.py                # Optional CLI version
│
├── uploads/               # Stores user-uploaded audio
├── output/                # Stores results: wav + json
│
├── templates/
│   └── index.html         # Web UI
│
├── static/
│   └── style.css          # Styling
│
├── venv310/               # Python 3.10 virtual environment
├── README.md
└── requirements.txt
```

---

## 🧑‍💻 Installation & Setup

### 1️⃣ Install Python 3.10  
Required because audio libraries do not support Python 3.13+ yet.

### 2️⃣ Create and Activate Virtual Environment

```bash
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
```

### 3️⃣ Install Dependencies

```bash
pip install flask librosa soundfile numpy
pip install torch==2.0.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## ▶️ Run the Flask Web App

Activate environment:

```bash
.\venv310\Scripts\Activate.ps1
```

Run:

```bash
python app.py
```

Then open browser:

```
http://127.0.0.1:5000/
```

---

## 🎧 Usage

1. **Upload Target Speaker Audio**  
2. **Upload Mixture Audio**  
3. Click **Process**  
4. Download:  
   - 🎧 `target_speaker.wav`  
   - 🧾 `diarization.json`

---

## 📄 Sample Output (JSON)

```json
[
  {
    "speaker": "Target",
    "start": 0.0,
    "end": 1.0,
    "similarity": 0.93
  },
  {
    "speaker": "Other",
    "start": 1.0,
    "end": 2.0,
    "similarity": 0.40
  }
]
```

---

## 🧠 How It Works (Simplified)

1. **Voice Activity Detection (VAD)** removes silence  
2. Mixture audio is **split into small chunks**  
3. For each chunk, compute:
   - Energy  
   - Amplitude  
   - Zero-crossing rate  
4. Compare chunk embedding with target embedding  
5. Classify chunk as:
   - **Target Speaker**, or  
   - **Other Speaker**  
6. Concatenate target chunks → **final extracted audio**  
7. Save diarization metadata → **JSON**

---

## 📝 Requirements

- Python 3.10  
- Flask  
- Librosa  
- NumPy  
- SoundFile  
- PyTorch  
- Torchaudio  

---

## 🤝 Acknowledgements

- **Librosa** for audio analysis  
- **Flask** for the web interface  
- **NumPy** for processing  
- **PyTorch** for backend models  

---

## ⭐ If this project helped you…

Please ⭐ the repository on GitHub!  
It motivates further development 😊
