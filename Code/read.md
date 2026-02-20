# Indent Tracking – Fatigue Experiment Analysis

Python/OpenCV script to track nano/micro-indents over multiple days and detect fatigue stages via intensity changes + K-Means clustering.

Three versions exist — pick the one that matches your data best.

## Version Comparison

| Version | Time Handling                          | Session Structure                  | Best For                                                                 | Main Files / Output                     |
|---------|----------------------------------------|------------------------------------|--------------------------------------------------------------------------|-----------------------------------------|
| **v1**  | Fixed 0.25 h/frame (hard-coded)        | One prefix per day                 | Clean, regular recordings (~4 frames/hour, no gaps/splits)              | `clustering_result*.png`                |
| **v2**  | Accurate time = hours / frame_count    | Multiple parts per day supported   | Irregular frame counts, split sessions (part1/part2), real experiments   | `clustering_result*_1.png`, etc.        |
| **v3**  | Same as v1 (0.25 h/frame)              | **Single indent only**             | Quick test on one indent, debugging tracking, or very sparse data       | `clustering_result.png` (no _1 suffix)  |

### Quick Decision Guide

- My data is **clean, continuous, roughly same # frames every day**  
  → Use **Version 1** (simplest)

- My recordings are **split across folders/files**, different frame counts per day, or irregular sampling  
  → Use **Version 2** (most accurate time axis, production-ready)

- I just want to **quickly test tracking on one indent**, debug stitching/alignment, or data is very sparse  
  → Use **Version 3** (single-indent version, fewer decisions)

## Quick Start (any version)

1. Put images in the folder (or change `SEARCH_DIR`)
2. Adjust `SEGMENT_CONFIG` prefixes and start hours
3. Set correct `FILE_EXTENSION`
4. Run the script → select template → click indent(s) → wait
5. Check `clustering_result*.png` and `verification_grid*.png`

## Tuning Tips (all versions)

- `radius=8` in `get_indent_intensity()` → match ~half your indent diameter
- `search_window=80→120` → increase if specimen drifts a lot
- `threshold=0.35` in template matching → lower = more candidates
- `n_clusters=2` → try 3 if you see three clear fatigue stages

Good luck with your fatigue experiments!  
Questions/PRs welcome.
