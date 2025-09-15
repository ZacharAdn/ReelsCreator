# Quick Test Commands - Updated 2025-09-10

## 🚨 CRITICAL TRANSCRIPTION ISSUES IDENTIFIED:

### ❌ BROKEN MODELS (hanging/poor quality):
- `tiny` - Produces garbage "עוד עוד עוד" repetition
- `ivrit-v2-d4` - Hangs during transcription (CPU intensive)
- `large-v3-turbo` - Hangs during transcription 
- `medium` - Hangs for long videos (9+ minutes)

### ✅ WORKING SOLUTIONS:

# 1. SHORT VIDEOS ONLY - Test with small file first
python -m src data/IMG_4225.MP4 --transcription-model small --force-cpu --save-stage-outputs

# 2. DRAFT MODE - Fastest, bypasses problematic models
python -m src data/IMG_4262.MOV --profile draft --save-stage-outputs

# 3. BASE MODEL - More reliable for Hebrew
python -m src data/IMG_4225.MP4 --transcription-model base --force-cpu --save-stage-outputs

## ✅ FIXED: DATE-BASED RESULTS FOLDERS
# Now automatically creates results/2025-09-10/ structure

## Model Testing Commands:
python -m src --list-models

# Test with different segment sizes
python -m src data/IMG_4225.MP4 --segment-duration 60 --overlap-duration 15 --profile draft

# Force CPU processing (required for stability)
python -m src data/IMG_4225.MP4 --force-cpu --save-stage-outputs

## Recent Major Updates:
- ✅ Fixed date-based results folder structure (results/YYYY-MM-DD/)
- ❌ Transcription quality issues identified with larger models
- ⚠️ Hebrew models require significant CPU resources
- ✅ Pipeline architecture working (6 stages complete)
- ✅ Stage output debugging available with --save-stage-outputs