# Meme Poser

Strike a pose in front of your webcam and get the meme that matches it.

A gallery strip at the bottom of the camera window shows every meme pose you can try. The pose you match lights up, and poses you've found get a check mark.

Hold a pose for half a second to reveal its meme. While you hold it, the screen says "HOLD IT..." and a bar fills up. The meme stays until you clearly break the pose.

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
2. Write a scoring function in `pose_detection.py` and add it to `DETECTORS` under the same name (`"dab": score_dab`). It gets a `Body` and returns a score from 0 (not this pose) to 1 (clearly this pose). Measure distances with `body.dist()` and `body.above()`, which use shoulder widths, so the pose works at any distance from the camera. See `score_hands_behind_head` for an example. Until you add a function, the meme shows in the gallery but can't be matched.

## Tuning

The settings at the top of `pose_detection.py` control matching: `HOLD_SECONDS` (how long to hold), `ENTER_SCORE` / `EXIT_SCORE` (how strict), `MIN_VISIBILITY` and `SMOOTHING`.

## Tests

```powershell
python -m unittest discover tests
```

The tests use fake body positions, so they don't need a webcam.
