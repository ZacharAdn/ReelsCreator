#!/usr/bin/env python3
"""Download and install FFmpeg for this project"""

import os
import sys
import urllib.request
import zipfile
import shutil

def download_ffmpeg():
    """Download FFmpeg static build for Windows"""
    print("Downloading FFmpeg...")

    # FFmpeg essentials build (smaller download)
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = "ffmpeg.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {zip_path}")

        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("ffmpeg_temp")

        # Find the ffmpeg.exe in the extracted files
        for root, dirs, files in os.walk("ffmpeg_temp"):
            if "ffmpeg.exe" in files:
                ffmpeg_path = os.path.join(root, "ffmpeg.exe")
                ffprobe_path = os.path.join(root, "ffprobe.exe")

                # Create bin directory
                os.makedirs("bin", exist_ok=True)

                # Copy ffmpeg and ffprobe
                shutil.copy(ffmpeg_path, "bin/ffmpeg.exe")
                shutil.copy(ffprobe_path, "bin/ffprobe.exe")

                print("✅ FFmpeg installed to bin/ffmpeg.exe")
                break

        # Clean up
        os.remove(zip_path)
        shutil.rmtree("ffmpeg_temp")

        # Add bin to PATH for this session
        current_path = os.environ.get('PATH', '')
        bin_path = os.path.abspath('bin')
        if bin_path not in current_path:
            os.environ['PATH'] = f"{bin_path};{current_path}"

        print(f"\n✅ FFmpeg ready! Added {bin_path} to PATH")
        print("\nTo use FFmpeg in future sessions, add this to your PATH:")
        print(f"  {bin_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_ffmpeg()
