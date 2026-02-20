# 🔬 Cantilever Fatigue — Indent Analysis Toolkit

> **A collection of Python tools for tracking and quantifying surface indent changes on cantilever specimens under cyclic fatigue loading, across two complementary imaging modalities.**

---

## 🧭 Which Tool Do I Need?

This repo contains **two separate analysis pipelines** depending on how your images were captured:

| | **Indent Intensity Tracker** | **Heatmap Analysis** |
|---|---|---|
| **Image source** | Time-lapse camera / video frames from cantilever fatigue rig | High-resolution microscope images captured at discrete time points |
| **Image type** | Hundreds of `.png` frames per session, lower magnification, subtle contrast | Small set of `.tif` files (typically 4), high magnification, indent geometry clearly visible |
| **What it measures** | How pixel intensity at each indent changes continuously over time | How much the surface changed between time points, or how Weber contrast at each indent evolves |
| **Output** | Intensity-vs-time scatter plots, K-Means fatigue stage clustering | Inferno difference heatmaps, Weber contrast line plots |
| **Scripts** | `cantilever_tracker_single.py`, `cantilever_tracker_v1.py`, `cantilever_tracker_v2.py` | `heatmap_diff_map.py`, `heatmap_weber_contrast.py` |

---

## 📁 Repository Structure

```
cantilever-fatigue-tracker/
│
├── README.md
├── .gitignore
│
├── code/
│   ├── indent_intensity_tracker/
│   │   ├── cantilever_tracker_single.py    # Track ONE indent across time-lapse frames
│   │   ├── cantilever_tracker_v1.py        # Track MULTIPLE indents (Nov-style filenames)
│   │   └── cantilever_tracker_v2.py        # Track MULTIPLE indents (debanded, split sessions)
│   │
│   └── heatmap_analysis/
│       ├── heatmap_diff_map.py             # Pixel difference heatmap between time points
│       └── heatmap_weber_contrast.py       # Weber contrast per indent across time points
│
├── templates/
│   ├── indent_template.png                 # Auto-saved on first run (tracker scripts)
│   └── indent_template_1.png              # Auto-saved on first run (v2)
│
├── sample_outputs/
│   ├── indent_intensity_tracker/
│   │   ├── verification_grid.png
│   │   ├── clustering_result.png
│   │   ├── clustering_result_combined.png
│   │   └── clustering_result_individual.png
│   │
│   └── heatmap_analysis/
│       ├── diff_map_24h.png
│       ├── diff_map_48h.png
│       ├── diff_map_72h.png
│       └── combined_contrast_plot.png
│
└── .gitignore
```

---

## ⚙️ Setup

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

Python 3.8+ recommended. All scripts are self-contained — no additional packages needed beyond the above.

---
---

# PART 1 — Indent Intensity Tracker

## 📷 What Kind of Images Does This Work With?

This pipeline processes **time-lapse image sequences** captured from a fixed (or nearly fixed) camera during a fatigue test. Think of it as a camera watching the specimen surface continuously over days.

The images are typically:
- Extracted video frames or interval-captured stills, named sequentially
- Hundreds to thousands of images per recording session
- Lower magnification — the full grid of indents is visible in each frame as subtle dark spots
- Indent contrast changes gradually and subtly as the material accumulates fatigue damage

The pipeline tracks each indent's pixel intensity across every frame and plots how it evolves over the full experiment (e.g. 0–72 hours continuously).

---

## 🧪 When Is This Useful?

- You are fatigue-testing a cantilever beam and recording the surface with a camera over time
- You want a **quantitative, time-resolved** record of surface change at specific indent locations
- You have multiple recording sessions (e.g. Day 1, Day 2, Day 3) that need stitching into one timeline
- You want to automatically classify frames into early-stage vs late-stage fatigue using clustering

---

## 🚀 Which Tracker Script Should I Use?

### `cantilever_tracker_single.py`
Track **one indent** and get a single intensity-vs-time curve. Best for a quick first test, or when you only care about one specific location on the specimen. The selection window shows all detected candidates — you click the one you want.

### `cantilever_tracker_v1.py`
Track **multiple indents simultaneously**. Use this when your image filenames follow the pattern `Prefix (N).png` — for example `Nov_17 (1).png`, `Nov_17 (2).png`, etc. You click as many indents as you want during setup; there is no hard limit.

### `cantilever_tracker_v2.py`
Track **multiple indents simultaneously**. Use this when your filenames end with `_1_debanded.png` and your sessions are split into parts (e.g. `cantilever_48H_part1`, `cantilever_48H_part2`). Designed for the debanded/processed image output from video extraction pipelines.

---

## 🔧 What To Change For Your Own Images

All changes are at the top of each script inside the `CONFIGURATION` block.

### Step 1 — Set your image folder
```python
SEARCH_DIR = r"E:\your\image\folder"   # v2: full path to your images
SEARCH_DIR = "."                        # v1 / single: same folder as the script
```

### Step 2 — Set your file extension
```python
FILE_EXTENSION = ".png"               # v1 and single
FILE_EXTENSION = "_1_debanded.png"    # v2
```

### Step 3 — Define your sessions
```python
# v1 and single — format: ("filename_prefix", start_hour)
SEGMENT_CONFIG = [
    ("Day1_frames", 0),    # images starting with "Day1_frames", from hour 0
    ("Day2_frames", 24),
    ("Day3_frames", 48),
]

# v2 — format: ("filename_prefix", start_hour, number_of_frames_in_session)
SEGMENT_CONFIG = [
    ("session1_part1", 0,  50),
    ("session1_part2", 0,  80),
    ("session2",       24, 90),
]
```

### Step 4 — First run (interactive setup)
On first run a window opens asking you to:
1. **Draw a box** around one indent → saved as the template for future runs
2. **Click each indent** you want to track → press `C` to confirm

No limit on the number of indents you select. Individual and combined subplot grids auto-scale to however many you choose.

---

## 📊 Outputs — Indent Intensity Tracker

| File | Description |
|---|---|
| `verification_grid.png` | Snapshots of tracking circles on the specimen — always check this first |
| `clustering_result.png` | Single-indent: intensity over time coloured by K-Means cluster |
| `clustering_result_combined.png` | All indents on one chart, coloured by indent ID |
| `clustering_result_individual.png` | Grid of per-indent subplots |
| `indent_template.png` | Auto-saved template patch (delete to re-select) |

---
---

# PART 2 — Heatmap Analysis

## 🔬 What Kind of Images Does This Work With?

This pipeline is designed for **high-resolution microscope images** — optical or SEM — of the same specimen captured at discrete time points (typically after removing it from the fatigue rig).

The images are typically:
- A small set of files (e.g. 4 images: 0h, 24h, 48h, 72h)
- High bit-depth `.tif` format, high magnification
- Each indent is clearly resolved with visible facets, sharp edges, and well-defined geometry (Vickers, Berkovich, or similar)
- There may be slight misalignment between time points due to re-mounting the specimen

Because the indent geometry is sharp and crisp at this magnification, these scripts can use feature-based alignment (ORB keypoints) and compute meaningful contrast metrics directly from pixel intensities.

---

## 🧪 When Is This Useful?

- You removed the specimen from the fatigue rig at intervals and imaged it under a microscope
- You want to see **where on the surface** the most change occurred over time (spatial heatmap)
- You have a pre-made grid of reference indents and want to track how the **optical contrast of each one** evolves as the material deforms
- You want a semi-quantitative metric that does not require a profilometer

---

## 🚀 Which Heatmap Script Should I Use?

### `heatmap_diff_map.py`
Produces an **inferno-coloured difference heatmap** showing where the surface changed between your 0h reference and each subsequent time point. Bright regions = areas of greatest pixel change.

Best for spotting crack initiation zones, deformation bands, or regions of concentrated wear.

### `heatmap_weber_contrast.py`
Detects each indent in your grid using template matching, then computes **Weber Contrast** at every indent for every image. Plots contrast vs indent number with one line per time point, showing how the full grid evolves together.

Weber Contrast = `(Background Mean − Indent Mean) / Background Mean`

A rising contrast value over time means the indent is becoming darker relative to its surroundings — a signature of increasing surface damage around that location.

Best for: comparing how different parts of your grid respond to fatigue, and identifying which indents show the earliest or fastest contrast change.

---

## 🔧 What To Change For Your Own Images

### `heatmap_diff_map.py`

```python
# ⬇ Replace with your own filenames — files must be in the same folder as the script
FILENAME_REF = "your_sample_after_polish.tif"   # 0h reference image
FILENAME_24H = "your_sample_24hours.tif"
FILENAME_48H = "your_sample_48hours.tif"
FILENAME_72H = "your_sample_72hours.tif"
```

To add more time points (e.g. 96h):
1. Add `FILENAME_96H = "your_sample_96hours.tif"`
2. Add `OUT_DIFF_96 = "diff_map_96h.png"`
3. In `__main__`, add `aligned_96, diff_96 = align_and_diff(img_96h, img_0h, "96 Hours")`
4. Add a new `plt.subplot()` panel to the visualization

Key tunable — gamma contrast enhancement inside `align_and_diff()`:
```python
gamma = 0.6   # lower (e.g. 0.3) = amplify faint changes; higher (e.g. 1.0) = strong changes only
```

### `heatmap_weber_contrast.py`

```python
# ⬇ Full path to the folder containing your microscope images and template
IMAGE_DIR = r"C:\Users\YourName\microscope_images"

# ⬇ Your images in chronological order — add or remove lines to match your experiment
IMAGE_FILES = [
    "sample_after_polish.tif",    # 0h reference
    "sample_24hour.tif",
    "sample_48hour.tif",
    "sample_72hour.tif",
]

# ⬇ Match to your indent size and grid layout
INDENT_RADIUS = 50    # pixels — roughly the indent radius at your image magnification
GRID_SIZE = 25        # total indents: 16 for 4×4 grid, 25 for 5×5, 36 for 6×6
```

You also need an `indent_template.png` — a tight, square crop of **one representative indent** from your 0h reference image, saved into `IMAGE_DIR`. This is used by the template matching to locate all other indents automatically.

---

## 📊 Outputs — Heatmap Analysis

| File | Description |
|---|---|
| `diff_map_24h.png` | Inferno heatmap showing pixel change between 0h and 24h |
| `diff_map_48h.png` | Inferno heatmap: 0h vs 48h |
| `diff_map_72h.png` | Inferno heatmap: 0h vs 72h |
| `aligned_24h.tif` | 24h image after alignment to 0h reference (registered) |
| `verify_*.png` | Per-image overlay confirming which indents were detected |
| `combined_contrast_plot.png` | Weber contrast of all indents across all time points on one chart |

---
---

## 🧠 How Each Pipeline Works

### Indent Intensity Tracker

```
Time-lapse frames (hundreds per session)
        │
        ▼
Brightness normalization     ← corrects lighting drift between sessions
        │
        ▼
ECC image alignment          ← corrects small camera movement between frames
        │
        ▼
Template matching            ← locates each indent in every frame
        │
        ▼
Intensity sampling           ← measures mean pixel brightness in a circle at each indent
        │
        ▼
Session stitching            ← removes intensity jumps at day boundaries
        │
        ▼
K-Means clustering           ← groups data into 2 fatigue stages per indent
        │
        ▼
Plots + verification grid
```

### Heatmap Analysis

```
High-res microscope images (4 time points)
        │
        ├── heatmap_diff_map.py
        │       │
        │   ORB keypoint detection + Homography alignment
        │       │
        │   Absolute pixel difference → gamma contrast enhancement
        │       │
        │   Inferno heatmap saved per time point
        │
        └── heatmap_weber_contrast.py
                │
            Multi-scale template matching → detect full indent grid
                │
            Weber contrast calculated per indent per image
                │
            Line plot: contrast vs indent number, one line per time point
```

---

## 💡 Tips For Best Results

**Indent Intensity Tracker:**
- Use fixed, diffuse lighting throughout the experiment. Moving shadows add noise.
- Always inspect `verification_grid.png` before trusting the clustering output — tracking circles should sit cleanly on each indent across all snapshots.
- If tracking drifts over time, try reducing `search_window` or increasing ECC alignment iterations.
- Delete `indent_template.png` and re-run if a new specimen or magnification is used.

**Heatmap Analysis:**
- Take the 0h reference image immediately after polishing and before any loading begins.
- Re-mount the specimen in the same orientation each session — this helps ORB alignment converge faster and with less distortion.
- If `heatmap_diff_map.py` produces a noisy heatmap, increase the RANSAC threshold (e.g. from 5.0 to 10.0) inside `align_and_diff()`.
- For `heatmap_weber_contrast.py`, make sure `INDENT_RADIUS` doesn't overlap neighbouring indents. A safe rule of thumb: use half the centre-to-centre spacing of your indent grid.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Issues / Contributions

If you encounter alignment failures, template misses on unusual indent geometries, or want to extend the pipeline to 3D profilometry or SEM data, feel free to open an issue or pull request.
