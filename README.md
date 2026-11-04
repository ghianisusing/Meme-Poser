# Meme Poser

Strike a pose in front of your webcam and get the meme that matches it.

A gallery strip at the bottom of the camera window shows every meme pose you can try. The pose you match lights up, and poses you've found get a check mark.

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

## Controls

| Key | Action |
| --- | --- |
| `g` | Open or close the full gallery of all poses |
| `q` | Quit |

## Adding a meme

1. Put the image in `poses/`, e.g. `poses/dab.png`. The file name becomes the pose name, and it shows up in the gallery automatically.
2. Write a detector function in `pose_detection.py` and add it to `DETECTORS` under the same name (`"dab": detect_dab`). Until then the meme shows in the gallery but can't be matched.
