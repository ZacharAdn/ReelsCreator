#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple script to process all videos one by one"""

import os
import sys
import subprocess

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

PYTHON = r"reels_extractor_env\Scripts\python.exe"
SCRIPT = r"src\quick scripts\transcribe_advanced.py"

videos = [
    r"nice data\IMG_4252.MOV",
    r"nice data\IMG_4256.MOV",
    r"nice data\IMG_4262.MOV",
    r"nice data\IMG_4263.MOV"
]

for i, video in enumerate(videos, 1):
    print(f"\n{'='*80}")
    print(f"Processing video {i}/{len(videos)}: {os.path.basename(video)}")
    print('='*80)

    env = os.environ.copy()
    env['AUTO_VIDEO_PATH'] = video

    try:
        result = subprocess.run(
            [PYTHON, SCRIPT],
            env=env,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"\nVideo {i} completed successfully!")
        else:
            print(f"\nVideo {i} failed with return code {result.returncode}")
    except Exception as e:
        print(f"\nError: {e}")

print(f"\n{'='*80}")
print("All transcriptions complete!")
print('='*80)
