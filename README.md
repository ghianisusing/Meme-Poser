# Meme Poser

Strike a pose in front of your webcam and get the meme that matches it.

A gallery strip at the bottom of the camera window shows every meme pose you can try. The pose you match lights up, and poses you've found get a check mark.

Hold a pose for half a second to reveal its meme. While you hold it, the screen says "HOLD IT..." and a bar fills up. The meme then pops into the top-right corner of the camera view and stays until you clearly break the pose. Press space while it's showing to save a photo of you next to the meme.

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
| `Space` | Save a photo of you next to the meme (while a meme is showing) to `snapshots/` |
| `s` | Show or hide your skeleton |
| `r` | Record a pose (then press the pose's number) |
| `Esc` | Cancel recording |
| `q` | Quit |

## Adding a meme

1. Put the image in `poses/`, e.g. `poses/dab.png`. The file name becomes the pose name, and it shows up in the gallery automatically, tagged "press r to set up".
2. Run the game, press `r`, then the meme's number. You get 3 seconds to get into the pose, then the game records you for 3 seconds. Keep your face and both arms in view.

The examples are saved to `poses/samples/dab.npy`. From then on the game recognizes the pose by comparing you to those examples. Recording again adds more examples, which helps if the pose is hard to trigger. Delete the `.npy` file to start over.

You can also write a scoring rule by hand for more control: add a function to `DETECTORS` in `detection.py` under the same name (`"dab": score_dab`). It gets a `Body` and returns a score from 0 (not this pose) to 1 (clearly this pose). See `score_hands_behind_head` for an example. If a pose has both a rule and recorded examples, the higher score counts.

## Checking your poses

Once you've recorded a few memes, check whether the game mixes any of them up:

```powershell
python check_poses.py
```

For each recorded pose it shows how often the game recognizes it, which other poses it could be mistaken for, and what to do about it:

```
Pose               Examples  Recognized  Mixed up with
-----------------  --------  ----------  -------------
arms_down                12     too few  -
surprised_guy            40          0%  hands_behind_head (100%)
hands_behind_head        92        100%  surprised_guy (100%)

To improve:
- arms_down: only 12 examples: record it again (press r)
- surprised_guy: often looks like hands_behind_head: make the two poses more different, or record both again
```

Two memes with nearly the same pose will always get mixed up. Pick memes whose poses look clearly different.

## Tuning

All settings are in `settings.py`: `HOLD_SECONDS` (how long to hold), `ENTER_SCORE` / `EXIT_SCORE` (how strict), `MIN_VISIBILITY`, `SMOOTHING`, the record and reveal timings, and `SAMPLE_NEAR` / `SAMPLE_FAR` (how closely you need to copy a recorded pose).

## Project layout

| File | What it does |
| --- | --- |
| `pose_detection.py` | Runs the game: camera, keys, and switching between play and record mode |
| `settings.py` | Every setting you can tune |
| `detection.py` | Hand-written pose rules and the hold timer |
| `pose_samples.py` | Recorded pose examples and matching against them |
| `library.py` | Loads the meme images from `poses/` |
| `ui.py` | Everything drawn on screen, plus snapshot photos |
| `check_poses.py` | Reports which recorded poses the game mixes up |

## Tests

```powershell
python -m unittest discover tests
```

The tests use fake body positions, so they don't need a webcam.
