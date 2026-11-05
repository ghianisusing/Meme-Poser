"""Mix-up check: which poses does the game confuse with each other?

Run: python check_poses.py

For every recorded example of every pose, it asks "if a player did exactly
this, which meme would the game pick?" and reports:
- how often each pose is recognized as itself
- which other poses it could be mistaken for
- which poses need more recordings

Hand-written pose rules (detection.DETECTORS) are checked too: a recorded
pose that accidentally triggers a rule shows up as a mix-up.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import pose_samples
from detection import DETECTORS, Body
from settings import ENTER_SCORE, EXIT_SCORE, SAMPLES_DIR

# Examples recorded right next to each other are almost identical, so they're
# left out when scoring an example against its own pose. Otherwise every pose
# would look perfect. At ~30 frames per second this skips about half a second.
NEIGHBOR_WINDOW = 15
MIN_EXAMPLES = 30  # suggest recording more below this
GOOD_RATE = 0.9  # recognized at least this often counts as fine


@dataclass
class PoseResult:
    name: str
    examples: int
    recognized: float | None  # share of examples the game would show this pose for (None = too few to check)
    own_score: float  # average score against its own other examples
    mixups: dict = field(default_factory=dict)  # other pose -> share of examples that score >= EXIT_SCORE


def to_landmarks(vec):
    """Turn a recorded example back into landmarks so hand-written rules can score it.
    Ears and other points weren't recorded, so they're marked as not visible."""
    lm = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for (x, y), idx in zip(vec, pose_samples.POINTS):
        # Recorded points are in shoulder widths around the shoulder center; place them
        # in a 1x1 "frame" (any scale works, the rules measure in shoulder widths too).
        lm[idx] = SimpleNamespace(x=0.5 + x * 0.1, y=0.5 + y * 0.1, visibility=1.0)
    return lm


def score_example(vec, index, own_name, samples, detectors):
    """Scores for every pose, for one recorded example of `own_name`."""
    scores = {}
    body = Body(to_landmarks(vec), 1, 1)
    for name, detector in detectors.items():
        scores[name] = detector(body)
    for name, examples in samples.items():
        if name == own_name:
            keep = np.ones(len(examples), bool)
            keep[max(0, index - NEIGHBOR_WINDOW) : index + NEIGHBOR_WINDOW + 1] = False
            examples = examples[keep]
        scores[name] = max(scores.get(name, 0.0), pose_samples.score(vec, examples))
    return scores


def check(samples, detectors=None):
    """Check every recorded pose. Returns a list of PoseResult."""
    detectors = DETECTORS if detectors is None else detectors
    results = []
    for name, examples in samples.items():
        hits, own_total = 0, 0.0
        mixup_counts = {}
        # With a rule for this pose, every example can be checked against the rule.
        # Without one, an example needs other examples far enough away to compare with.
        checkable = name in detectors or len(examples) > 2 * NEIGHBOR_WINDOW + 1
        for i, vec in enumerate(examples):
            scores = score_example(vec, i, name, samples, detectors)
            own = scores.get(name, 0.0)
            own_total += own
            best = max(scores, key=scores.get)
            if best == name and own >= ENTER_SCORE:
                hits += 1
            for other, s in scores.items():
                if other != name and s >= EXIT_SCORE:
                    mixup_counts[other] = mixup_counts.get(other, 0) + 1
        n = len(examples)
        results.append(
            PoseResult(
                name=name,
                examples=n,
                recognized=(hits / n if n else 0.0) if checkable else None,
                own_score=own_total / n if n else 0.0,
                mixups={k: v / n for k, v in sorted(mixup_counts.items(), key=lambda kv: -kv[1])},
            )
        )
    return results


def advice(r):
    tips = []
    if r.examples < MIN_EXAMPLES:
        tips.append(f"only {r.examples} examples: record it again (press r)")
    if r.mixups:
        worst, share = next(iter(r.mixups.items()))
        if share >= 0.1:
            tips.append(f"often looks like {worst}: make the two poses more different, or record both again")
    if r.recognized is not None and r.recognized < GOOD_RATE and not tips:
        tips.append("not recognized reliably: record it again, holding the pose steadily")
    return tips


def format_report(results, detectors=None):
    detectors = DETECTORS if detectors is None else detectors
    if not results:
        return (
            "No recorded poses yet.\n"
            "Run the game, press r, and record a pose first. Recordings are saved in poses/samples/."
        )

    lines = []
    width = max(len(r.name) for r in results)
    lines.append(f"{'Pose':<{width}}  Examples  Recognized  Mixed up with")
    lines.append(f"{'-' * width}  --------  ----------  -------------")
    for r in sorted(results, key=lambda r: -1 if r.recognized is None else r.recognized):
        mix = ", ".join(f"{k} ({v:.0%})" for k, v in r.mixups.items() if v >= 0.01) or "-"
        rec = "too few" if r.recognized is None else f"{r.recognized:.0%}"
        lines.append(f"{r.name:<{width}}  {r.examples:>8}  {rec:>10}  {mix}")

    lines.append("")
    problems = [(r, advice(r)) for r in results]
    problems = [(r, tips) for r, tips in problems if tips]
    if problems:
        lines.append("To improve:")
        for r, tips in problems:
            for tip in tips:
                lines.append(f"- {r.name}: {tip}")
    else:
        lines.append("All recorded poses look fine.")

    rule_only = [name for name in detectors if name not in {r.name for r in results}]
    if rule_only:
        lines.append("")
        lines.append(
            "Poses with a hand-written rule and no recordings aren't checked on their own: "
            + ", ".join(rule_only)
            + ". They still show up above if a recorded pose triggers them."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check which recorded poses the game mixes up.")
    parser.add_argument("--samples", default=SAMPLES_DIR, help="folder with recorded poses (default: poses/samples)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.samples):
        print(format_report([]))
        return 0
    print(format_report(check(pose_samples.load_samples(args.samples))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
