# Meme Poser

Copy the meme pose in front of your webcam.

## Setup (Windows)

Needs Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

## Run

```powershell
python pose_detection.py   # pose matcher
python face_detector.py    # face detection demo
```

Press `q` to quit.
