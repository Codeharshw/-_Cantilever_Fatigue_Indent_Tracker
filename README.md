# 🔬 Cantilever Fatigue Indent Tracker

> **Automated optical tracking and contrast-intensity analysis of micro-indents on cantilever specimens under cyclic fatigue loading.**

---

## 📌 What This Tool Does

This pipeline processes time-lapse microscopy or camera images of a **cantilever beam specimen** that has been pre-marked with a grid of micro-indents. As the specimen undergoes **fatigue loading over time (e.g., 0–72 hours)**, the shape and contrast of each indent changes subtly due to material deformation and surface damage accumulation.

This tool:
- Automatically **tracks** each indent across hundreds of frames using template matching
- **Measures pixel intensity** at each indent location as a proxy for indent depth/contrast change
- **Stitches** multi-session data (e.g., Day 1, Day 2, Day 3) into a continuous timeline
- Applies **K-Means clustering** to classify measurement points into early-stage vs. late-stage fatigue behaviour
- Produces **publication-ready plots** — both a combined overview and individual per-indent charts

### Example: Tracking Verification Grid

The image below shows 25 indents being tracked across multiple time points. Red circles (with green labels) indicate active tracking; open circles show tracked positions in later frames as the specimen surface evolves.

> *(Insert your `verification_grid.png` here — generated automatically after running the script)*

---

## 🧪 When Is This Useful?

This code is specifically designed for experimental scenarios where:

| Scenario | Description |
|---|---|
| **Fatigue testing** | You are cycling a specimen mechanically and want to monitor surface damage non-destructively over time |
| **Creep/relaxation studies** | Long-duration loading where indent geometry changes slowly |
| **Optical profilometry alternatives** | When you have a camera but not a profilometer, and want a semi-quantitative indent depth proxy |
| **Multi-session experiments** | Your data is recorded across multiple days/sessions with slight camera repositioning between sessions |
| **Grid indentation arrays** | You have a regular or irregular grid of reference indents (Vickers, Berkovich, or custom) |
| **SEM/optical microscopy time-lapse** | Any image sequence where surface features must be tracked across many frames |

**Typical applications include:**
- Metallic cantilever beams under bending fatigue
- Polymer creep specimens
- Thin film delamination tracking
- Bone/biomaterial fatigue studies
- Any specimen where visible surface marks serve as deformation indicators

---

## 📁 Repository Structure

```
cantilever-fatigue-tracker/
│
├── cantilever_tracker_v1.py        # Version 1 — single-day, basic pipeline
├── cantilever_tracker_v2.py        # Version 2 — multi-session, debanded images
│
├── indent_template.png             # Auto-saved when you select a template (v1)
├── indent_template_1.png           # Auto-saved when you select a template (v2)
│
├── outputs/
│   ├── clustering_result_combined.png      # Combined plot of all indents
│   ├── clustering_result_individual.png    # Per-indent subplot grid
│   └── verification_grid.png              # Tracking accuracy visual check
│
└── README.md
```

---

## ⚙️ Setup & Requirements

### Install dependencies

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### Python version
Python 3.8+ recommended.

---

## 🚀 How To Run

```bash
python cantilever_tracker_v2.py
```

On first run, an interactive window will open asking you to:
1. **Select a template** — draw a box around one indent (this is saved for future runs)
2. **Click to select indents** — click near each indent you want to track, press `C` to confirm

---

## 🔧 Where To Make Changes For Your Own Images

All configuration is at the top of the script. **You only need to edit the blocks marked below.**

---

### ✏️ 1. Set Your Folder Path

```python
# -----------------------------------------------
# ⬇ CHANGE THIS to the folder containing your images
# -----------------------------------------------
SEARCH_DIR = r"E:\CANTILEVER EXPERIMENT VIDEOS\03-02-2026\extracted_frames\research_processed_results"
```

Replace this with the absolute path to your image folder, e.g.:
- Windows: `r"C:\Users\YourName\experiment_images"`
- Mac/Linux: `"/home/yourname/experiment_images"`

---

### ✏️ 2. Set Your File Extension / Suffix

```python
# -----------------------------------------------
# ⬇ CHANGE THIS to match your image filename ending
# -----------------------------------------------
FILE_EXTENSION = "_1_debanded.png"
```

This filters which files in your folder are loaded. If your images are plain `.png` or `.jpg`, change to:
```python
FILE_EXTENSION = ".png"
# or
FILE_EXTENSION = ".jpg"
```

---

### ✏️ 3. Define Your Time Segments (Sessions/Days)

```python
# -----------------------------------------------
# ⬇ CHANGE THIS to match your experimental sessions
# Each entry: ("filename_prefix", start_hour_of_session, number_of_frames_in_session)
# -----------------------------------------------
SEGMENT_CONFIG = [
    ("24_hour_data_1",      0,  19),   # Session 1: 0–24 hrs, 19 frames
    ("24_hour_data_2",      0,  38),   # Session 1 alt: 0–24 hrs, 38 frames
    ("cantilever_48H_part1", 24, 3),   # Session 2: 24–48 hrs, 3 frames
    ("cantilever_48H_part2", 24, 85),  # Session 2 cont.: 24–48 hrs, 85 frames
    ("72_hour_data_part1",  48, 6),    # Session 3: 48–72 hrs, 6 frames
    ("72_hour_data_part2",  48, 90)    # Session 3 cont.: 48–72 hrs, 90 frames
]
```

**Each tuple contains:**
- `"prefix"` — the start of your image filenames for that session
- `start_hour` — what clock hour does this session begin at
- `max_frames` — total number of frames expected (used for time interpolation)

**Example for a simple 3-day experiment:**
```python
SEGMENT_CONFIG = [
    ("day1_images", 0,  100),   # Day 1: frames named day1_images_XXXXXX.png
    ("day2_images", 24, 100),   # Day 2
    ("day3_images", 48, 100),   # Day 3
]
```

---

### ✏️ 4. Set the Template Filename

```python
# -----------------------------------------------
# ⬇ CHANGE THIS if you want a different template save name
# -----------------------------------------------
TEMPLATE_FILENAME = "indent_template_1.png"
```

Delete this file to force re-selection of a new template on the next run.

---

### ✏️ 5. (Optional) Tune Tracking Parameters

These are advanced settings you can adjust if tracking quality is poor:

```python
ALIGN_METHOD = "ECC"      # Image alignment method — keep as "ECC" for best results
REMOVE_ZEROS = True       # Filter out dropout frames
STITCH_DATA = True        # Smooth intensity jumps between sessions
VERIFY_SAMPLES = 15       # How many verification snapshots to save
```

Inside `detect_grid_candidates()`:
```python
threshold = 0.35   # Lower = find more candidates; raise if too many false detections
min_dist = 30      # Minimum pixel distance between detected indents
```

Inside `track_template_local()`:
```python
search_window = 80   # Pixel radius to search around last known position
```

---

## 📊 Output Files

| File | Description |
|---|---|
| `clustering_result_combined.png` | All indents on one plot, coloured by indent ID, clustered by fatigue stage |
| `clustering_result_individual.png` | Grid of per-indent plots |
| `verification_grid.png` | Visual check that tracking circles are landing on the correct indents |
| `indent_template_1.png` | Saved template patch (auto-generated) |

---

## 🧠 How The Analysis Works

```
Raw Images
    │
    ▼
Brightness Normalization  ←── Corrects for lighting drift between sessions
    │
    ▼
ECC Image Alignment       ←── Corrects for camera repositioning between frames
    │
    ▼
Template Matching         ←── Finds and tracks each indent's exact pixel location
    │
    ▼
Intensity Sampling        ←── Measures mean pixel brightness in a circular ROI at each indent
    │
    ▼
Session Stitching         ←── Removes intensity jumps at day boundaries
    │
    ▼
K-Means Clustering        ←── Separates data into 2 fatigue stages per indent
    │
    ▼
Plots & Verification Grid
```

The **intensity value** at each indent is a proxy for indent contrast. As a specimen fatigues, surface damage (slip bands, cracking, plastic flow) alters the reflectivity around indents — this change is what the clustering analysis captures.

---

## 💡 Tips For Best Results

- **Lighting consistency**: Use fixed, diffuse illumination. Avoid shadows or specular reflections moving between frames.
- **Frame rate**: More frames = smoother data. Even 1 frame per 15 minutes is sufficient.
- **Template selection**: Choose an indent that is sharp, isolated, and representative of others in the grid.
- **Verification grid**: Always inspect `verification_grid.png` before trusting the clustering output — if circles are drifting off the indents, reduce `search_window` or improve alignment.
- **Stitching**: If intensity jumps at session boundaries remain visible, inspect the overlap frames manually.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙋 Contributing / Issues

If you encounter issues with template matching on unusual indent geometries, or want to extend this to 3D profilometry data, feel free to open an issue or pull request.
