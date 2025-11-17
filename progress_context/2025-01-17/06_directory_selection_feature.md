# Feature: Directory-Based Video Selection for cut_video_segments.py

**Date**: 2025-01-17
**Type**: Feature Enhancement
**Status**: Completed

## Problem Statement

The `cut_video_segments.py` script only scanned the hardcoded `data/` directory for videos, while `transcribe_advanced.py` had a more flexible directory-based selection system that:
- Scans the entire project for directories containing videos
- Allows users to select from multiple directories
- Provides video counts per directory

User requested consistency between both scripts for better UX.

## User Request

> "אני רוצה שבחירת הוידאו פה תהיה כמו בסקריפט הראשי .....(לפי תקיות וכו)"
> Translation: "I want video selection here to be like in the main script (by folders, etc.)"

## Solution

Modified `cut_video_segments.py` to match the directory selection workflow from `transcribe_advanced.py`.

### Changes Made

#### 1. Renamed Function for Consistency
**Before**:
```python
def scan_video_directory(directory: str = "data") -> List[dict]:
```

**After**:
```python
def scan_directory_for_videos(directory: str) -> List[dict]:
```

- Removed default `"data"` parameter (now passed explicitly)
- Renamed to match transcription script naming

#### 2. Added Directory Discovery Function
**New function** (copied from `transcribe_advanced.py`):
```python
def find_directories_with_videos(base_path: str = ".") -> List[str]:
    """
    Find all directories in the project that contain video files

    Args:
        base_path: Base path to search from (default: current directory)

    Returns:
        List of directory paths containing videos
    """
    video_extensions = ('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV')
    directories_with_videos = []

    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories and common excludes
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', 'reels_extractor_env', '__pycache__']]

        # Check if this directory contains any video files
        has_videos = any(filename.endswith(video_extensions) for filename in files)

        if has_videos:
            directories_with_videos.append(root)

    return sorted(directories_with_videos)
```

#### 3. Updated interactive_mode() - Two-Step Selection

**Before** (single step - hardcoded data/):
```python
def interactive_mode():
    # Scan data directory for videos
    videos = scan_video_directory("data")

    if not videos:
        print("\n❌ No video files found in data/ directory")
        sys.exit(1)

    # Display video list and select...
```

**After** (two steps - directory then video):
```python
def interactive_mode():
    # STEP 1: Find and select directory
    print("\n🔍 Scanning project for video files...")
    video_dirs = find_directories_with_videos(".")

    if not video_dirs:
        print("\n❌ No directories with video files found in project")
        sys.exit(1)

    # Display directory options
    print("\n" + "=" * 80)
    print("📁 Directories with videos:")
    print("=" * 80)
    for i, directory in enumerate(video_dirs, 1):
        videos_in_dir = scan_directory_for_videos(directory)
        print(f"\n{i}. {directory}")
        print(f"   ({len(videos_in_dir)} video{'s' if len(videos_in_dir) != 1 else ''})")
    print("\n" + "=" * 80)

    # Get directory selection
    selection = input("> ").strip()
    # ... validation logic ...

    selected_dir = video_dirs[dir_index]
    print(f"\n✅ Selected directory: {selected_dir}")

    # STEP 2: Scan selected directory and select video
    videos = scan_directory_for_videos(selected_dir)

    # Display video list and select...
    print(f"\n✅ Selected: {videos[video_index]['name']}")
    print(f"📊 Duration: {format_time(videos[video_index]['duration'])} | Size: {videos[video_index]['size_mb']:.1f}MB")
```

## Testing Results

```bash
$ python src/scripts/cut_video_segments.py
================================================================================
VIDEO SEGMENT CUTTER - Interactive Mode
================================================================================

🔍 Scanning project for video files...

================================================================================
📁 Directories with videos:
================================================================================

1. ./data
   (6 videos)

2. ./data/latest
   (11 videos)

3. ./data/merav2
   (2 videos)

4. ./data/ran1
   (3 videos)

5. ./generated_data
   (10 videos)

================================================================================

Select directory number (or press Enter to cancel):
> 1

✅ Selected directory: ./data

================================================================================
📹 Available Videos
================================================================================

1. IMG_4225.MP4
   Duration: 3:38.91 | Size: 93.9MB | Date: 2025-08-08

2. IMG_4280.MOV
   Duration: 10:21.23 | Size: 586.3MB | Date: 2025-10-07

[... more videos ...]
```

## Benefits

1. **Consistency**: Both scripts now use identical video selection workflow
2. **Flexibility**: Users can work with videos in any directory (data/, data/latest/, generated_data/, etc.)
3. **Better UX**: Shows video counts per directory before selection
4. **Discoverability**: Automatically finds all video directories in the project

## Files Modified

- `src/scripts/cut_video_segments.py`:
  - Added `find_directories_with_videos()` (lines 394-418)
  - Renamed `scan_video_directory()` → `scan_directory_for_videos()` (lines 361-392)
  - Updated `interactive_mode()` with two-step selection (lines 438-514)

## Backward Compatibility

✅ **Command-line mode unchanged**: `--video` and `--ranges` arguments still work exactly as before
✅ **Interactive mode enhanced**: Now offers directory selection before video selection

## Related Files

- Reference implementation: `src/scripts/transcribe_advanced.py` (lines 376-433, 453-534)
- Documentation: `CLAUDE.md` (to be updated with new workflow)

## Impact

- **User Experience**: ⬆️ Improved (more flexible directory handling)
- **Code Consistency**: ⬆️ Improved (matches transcription script)
- **Breaking Changes**: ❌ None (command-line mode unchanged)
- **Performance**: ➡️ Neutral (same speed, just scans more directories initially)
