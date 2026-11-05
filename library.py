"""The meme images in the poses folder."""

import os

import cv2

from settings import IMAGE_EXTS


def display_name(name):
    return name.replace("_", " ").replace("-", " ").capitalize()


def load_poses(folder):
    """Load every image in the poses folder. Returns {name: image}."""
    poses = {}
    if not os.path.isdir(folder):
        return poses

    for file in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(os.path.join(folder, file))
        if image is None:
            print(f"Skipping pose image that could not be read: {file}")
            continue
        poses[stem] = image

    return poses
