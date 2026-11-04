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
| `r` | Record a pose (then press the pose's number) |
| `Esc` | Cancel recording |
| `q` | Quit |

## Adding a meme

1. Put the image in `poses/`, e.g. `poses/dab.png`. The file name becomes the pose name, and it shows up in the gallery automatically, tagged "press r to set up".
2. Run the game, press `r`, then the meme's number. You get 3 seconds to get into the pose, then the game records you for 3 seconds. Keep your face and both arms in view.

The examples are saved to `poses/samples/dab.npy`. From then on the game recognizes the pose by comparing you to those examples. Recording again adds more examples, which helps if the pose is hard to trigger. Delete the `.npy` file to start over.

You can also write a scoring rule by hand for more control: add a function to `DETECTORS` in `pose_detection.py` under the same name (`"dab": score_dab`). It gets a `Body` and returns a score from 0 (not this pose) to 1 (clearly this pose). See `score_hands_behind_head` for an example. If a pose has both a rule and recorded examples, the higher score counts.

## Tuning

The settings at the top of `pose_detection.py` control matching: `HOLD_SECONDS` (how long to hold), `ENTER_SCORE` / `EXIT_SCORE` (how strict), `MIN_VISIBILITY` and `SMOOTHING`, plus the record timings. For recorded poses, `NEAR` and `FAR` in `pose_samples.py` set how closely you need to copy the recording.

## Tests

```powershell
python -m unittest discover tests
```

The tests use fake body positions, so they don't need a webcam.
