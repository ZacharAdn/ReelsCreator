#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check transcription results"""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

RESULTS_DIR = "results"

videos = [
    "IMG_4252",
    "IMG_4256",
    "IMG_4262",
    "IMG_4263"
]

print("Checking transcription results...")
print("="*80)

if not os.path.exists(RESULTS_DIR):
    print(f"Results directory not found: {RESULTS_DIR}")
    sys.exit(1)

for video in videos:
    # Find directory for this video
    found = False
    for dirname in os.listdir(RESULTS_DIR):
        if video in dirname:
            result_dir = os.path.join(RESULTS_DIR, dirname)
            summary_file = os.path.join(result_dir, f"{video}_final_summary.txt")

            if os.path.exists(summary_file):
                # Get file size
                size = os.path.getsize(summary_file)
                print(f"✓ {video}: Transcribed ({size} bytes)")
                found = True
                break
            else:
                print(f"⧗ {video}: In progress (directory exists but no summary yet)")
                found = True
                break

    if not found:
        print(f"✗ {video}: Not started")

print("="*80)
