import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# ⬇ CHANGE THIS: Define your experimental sessions.
# Each tuple: ("image_filename_prefix", start_hour_of_session, number_of_frames_in_session)
#
# The prefix must exactly match the beginning of your image filenames.
# max_frames is used to interpolate time evenly across the session.
#
# Example for a simple 3-day run:
#   SEGMENT_CONFIG = [
#       ("day1_frames", 0,  100),
#       ("day2_frames", 24, 100),
#       ("day3_frames", 48, 100),
#   ]
#
# Split sessions (part1 / part2) are supported — just add extra entries
# with the same start_hour and different prefixes.
SEGMENT_CONFIG = [
    ("24_hour_data_1",      0,  19),   # 0–24 hrs, 19 frames
    ("24_hour_data_2",      0,  38),   # 0–24 hrs, 38 frames (alternative recording)
    ("cantilever_48H_part1", 24, 3),   # 24–48 hrs, 3 frames
    ("cantilever_48H_part2", 24, 85),  # 24–48 hrs, 85 frames
    ("72_hour_data_part1",  48, 6),    # 48–72 hrs, 6 frames
    ("72_hour_data_part2",  48, 90)    # 48–72 hrs, 90 frames
]

# ⬇ CHANGE THIS: The suffix/extension your image filenames end with.
# This filters which files in your folder are loaded.
# Common options: ".png", ".jpg", ".tif"
# For plain images with no suffix before extension: FILE_EXTENSION = ".png"
FILE_EXTENSION = "_1_debanded.png"

# ⬇ CHANGE THIS: Full path to the folder containing your images.
# Windows example: r"C:\Users\YourName\experiment_images"
# Mac/Linux example: "/home/yourname/experiment_images"
SEARCH_DIR = r"E:\CANTILEVER EXPERIMENT VIDEOS\03-02-2026\extracted_frames\research_processed_results"

# ⬇ CHANGE THIS (optional): Name of the saved template file.
# Auto-created on first run when you draw a box around one indent.
# Delete this file to force re-selection of a new template on next run.
TEMPLATE_FILENAME = "indent_template_1.png"

# --- SETTINGS ---
# These can be left as-is for most experiments. See inline notes if you need to tune.
ALIGN_METHOD = "ECC"     # Alignment method — keep as "ECC" for best accuracy
REMOVE_ZEROS = True      # Filter frames where intensity is zero (camera dropout)
STITCH_DATA = True       # Smooth intensity jumps between sessions — recommended ON
VERIFY_SAMPLES = 15      # Max verification snapshots to save — increase for more checkpoints

# Global variable for mouse interaction — do not change
selected_point = None

# ==========================================
# 0. BRIGHTNESS NORMALIZATION
# ==========================================

def normalize_brightness(img, ref_stats):
    """
    Adjusts the brightness/contrast of 'img' to match the mean/std-dev
    of the master reference image.
    This corrects for lighting drift between sessions — do not remove.
    """
    if len(img.shape) == 3:
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = img
        
    m_src, s_src = cv2.meanStdDev(src_gray)
    m_ref, s_ref = ref_stats
    
    if s_src[0][0] < 1e-5: return img
    
    gain = s_ref[0][0] / s_src[0][0]
    offset = m_ref[0][0] - (m_src[0][0] * gain)
    
    normalized = cv2.convertScaleAbs(img, alpha=gain, beta=offset)
    return normalized

# ==========================================
# 1. TEMPLATE MATCHING LOGIC
# ==========================================

def load_or_create_template(img_gray):
    """
    Loads template from file, or opens a selection window if not found.
    The template is a small image patch around ONE typical indent.
    It is used as the reference shape for finding and tracking all other indents.
    """
    if os.path.exists(TEMPLATE_FILENAME):
        template = cv2.imread(TEMPLATE_FILENAME, 0)
        
        if template is None:
            print(f"⚠ Found {TEMPLATE_FILENAME} but could not read it. Please re-select.")
        else:
            if template.shape[0] > img_gray.shape[0] // 2:
                template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
            return template
    
    print(f"\n⚠ '{TEMPLATE_FILENAME}' not found or invalid!")
    print("Please select a template area from the first image.")
    
    x, y, w, h = 0, 0, 0, 0
    try:
        r = cv2.selectROI("Select Indent Template", img_gray, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Indent Template")
        x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    except Exception as e:
        print(f"⚠ UI Selection failed: {e}")

    if w == 0 or h == 0:
        print("⚠ Selection invalid. Auto-cropping center.")
        img_h, img_w = img_gray.shape
        w, h = 50, 50
        x = max(0, img_w // 2 - w // 2)
        y = max(0, img_h // 2 - h // 2)
        
    template = img_gray[y:y+h, x:x+w]
    cv2.imwrite(TEMPLATE_FILENAME, template)
    return template

def detect_grid_candidates(img_gray, template):
    """
    Uses Multi-Scale Template Matching to find all potential indents automatically.
    
    ⬇ TUNE IF NEEDED:
    - threshold (line ~170): lower value (e.g. 0.25) finds more candidates,
      higher (e.g. 0.5) is stricter. Lower if indents are being missed.
    - min_dist (line ~183): minimum pixel separation between two detected indents.
      Increase if the same indent appears detected twice, decrease if nearby indents are merged.
    """
    if template is None: return []

    h, w = template.shape
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    scales = [0.9, 1.0, 1.1]
    all_matches = []
    
    print("   > Running multi-scale template matching...")
    
    for scale in scales:
        if scale != 1.0:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            curr_template = cv2.resize(template, (scaled_w, scaled_h))
        else:
            curr_template = template
            scaled_w, scaled_h = w, h
            
        res = cv2.matchTemplate(blurred, curr_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.35  # ⬅ TUNE: lower = more detections, higher = stricter
        locs = np.where(res >= threshold)
        
        for pt in zip(*locs[::-1]):
            center_x = pt[0] + scaled_w // 2
            center_y = pt[1] + scaled_h // 2
            score = res[pt[1], pt[0]]
            all_matches.append((center_x, center_y, score))

    all_matches.sort(key=lambda x: x[2], reverse=True)
    detected_centers = []
    min_dist = 30  # ⬅ TUNE: increase (e.g. 50) if same indent detected twice; decrease (e.g. 15) if nearby indents are merged
    
    for cx, cy, score in all_matches:
        is_duplicate = False
        for existing_x, existing_y in detected_centers:
            dist = np.sqrt((cx - existing_x)**2 + (cy - existing_y)**2)
            if dist < min_dist:
                is_duplicate = True
                break
        if not is_duplicate:
            detected_centers.append((cx, cy))
            
    print(f"   > Found {len(detected_centers)} potential indents.")
    return detected_centers

# ==========================================
# 2. INTERACTIVE SELECTION
# ==========================================

def get_tracking_template(img, x, y, size=40):
    """
    Extracts a small image patch around (x, y) to use as a per-indent tracking template.
    
    ⬇ TUNE IF NEEDED:
    - size=40: patch size in pixels. Increase (e.g. 60–80) for larger indents,
      decrease (e.g. 20) for very small indents. Should fully contain one indent.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    h, w = gray.shape
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    return gray[y1:y2, x1:x2]

def track_template_local(img, start_x, start_y, tracking_template, search_window=80):
    """
    Searches for the tracking template within a local window around the last known position.
    Returns the new center coordinates and a confidence score.
    
    ⬇ TUNE IF NEEDED:
    - search_window=80: search radius in pixels from last known position.
      Increase (e.g. 150) if the specimen moves significantly between frames.
      Decrease (e.g. 40) if nearby features are causing false matches.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    h, w = gray.shape
    t_h, t_w = tracking_template.shape
    
    x1 = max(0, start_x - search_window)
    y1 = max(0, start_y - search_window)
    x2 = min(w, start_x + search_window)
    y2 = min(h, start_y + search_window)
    
    roi = gray[y1:y2, x1:x2]
    
    if roi.shape[0] < t_h or roi.shape[1] < t_w:
        return start_x, start_y, 0.0
    
    res = cv2.matchTemplate(roi, tracking_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    match_center_x = x1 + max_loc[0] + t_w // 2
    match_center_y = y1 + max_loc[1] + t_h // 2
    
    return match_center_x, match_center_y, max_val

# Global state for click selection — do not change
last_click = None

def click_selection_callback(event, x, y, flags, param):
    global last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)

def select_indents_manual_roi(img, template, title_prefix="", pre_selected=None):
    """
    Opens an interactive window for you to manually click and select indents.

    Controls:
      - LEFT CLICK near an indent to select it (auto-snaps to exact center)
      - Press 'D' to delete the last selected indent
      - Press 'C' to confirm all selections and proceed
      - Press ESC to cancel (this session will be skipped)

    ℹ NOTE ON NUMBER OF INDENTS:
    There is NO hard-coded limit on how many indents you can select.
    - Fewer than 25: just click fewer points, then press C.
    - More than 25: keep clicking as many as you need, then press C.
    - The individual subplot grid auto-scales to the number you select.
    - The combined plot shows all of them, coloured by indent ID.
    """
    global last_click
    last_click = None
    
    print("\n" + "="*60)
    print(f"MANUAL INDENT SELECTION: {title_prefix}")
    print("="*60)
    print("1. CLICK near an indent to select it.")
    print("   (The script will snap to the exact center)")
    print("2. Press 'C' to confirm all selections.")
    print("3. Press 'D' (or Right Click) to delete the last selection.")
    print("="*60 + "\n")
    
    selected_indents = list(pre_selected) if pre_selected else []
    window_name = f"Select Indents: {title_prefix}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_selection_callback)
    
    th, tw = template.shape[:2]
    search_radius = 30  # ⬅ TUNE: pixel radius around your click to search for indent center
    
    while True:
        display_img = img.copy()
        
        for i, (sx, sy) in enumerate(selected_indents):
            color = (0, 0, 255)
            cv2.circle(display_img, (sx, sy), 15, color, 2)
            cv2.circle(display_img, (sx, sy), 2, color, -1)
            cv2.putText(display_img, f"#{i+1}", (sx+20, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        
        if last_click:
            mx, my = last_click
            last_click = None
            
            h, w = img.shape[:2]
            x1 = max(0, mx - search_radius)
            y1 = max(0, my - search_radius)
            x2 = min(w, mx + search_radius)
            y2 = min(h, my + search_radius)
            
            roi = img[y1:y2, x1:x2]
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
                
            if roi_gray.shape[0] >= th and roi_gray.shape[1] >= tw:
                try:
                    res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    center_x = x1 + max_loc[0] + tw // 2
                    center_y = y1 + max_loc[1] + th // 2
                    
                    is_duplicate = False
                    for existing in selected_indents:
                        if np.linalg.norm(np.array(existing) - np.array((center_x, center_y))) < 10:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        selected_indents.append((center_x, center_y))
                        print(f"   > Added Indent {len(selected_indents)} at ({center_x}, {center_y}) (Score: {max_val:.2f})")
                    else:
                        print("   > Indent already selected.")
                        
                except Exception as e:
                    print(f"   ⚠ Search error: {e}")
            else:
                print("   ⚠ Click too close to edge or ROI too small.")

        if key == ord('d'):
            if selected_indents:
                removed = selected_indents.pop()
                print(f"   > Removed last indent at {removed}")
                
        elif key == ord('c') or key == ord('C'):
            if selected_indents:
                cv2.destroyWindow(window_name)
                return selected_indents
            else:
                print("⚠ No indents selected! Please add at least one.")
                
        elif key == 27:
            cv2.destroyWindow(window_name)
            return []

# ==========================================
# 3. ANALYSIS UTILS
# ==========================================

def natural_sort_key(s):
    """Sorts filenames naturally (e.g. frame2 before frame10). Do not change."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_time_from_filename(filename, start_hour, max_frames):
    """
    Extracts the frame number from the filename and maps it to a time value.
    Currently expects filenames ending in: _XXXXXX_1_debanded.png
    where XXXXXX is a 6-digit frame number.

    ⬇ CHANGE THIS if your filenames use a different format.
    Examples:
      - Files named 'session1_frame001.png':
            match = re.search(r'frame(\d+)', filename)
      - Files named 'img_0042.tif':
            match = re.search(r'img_(\d+)', filename)
    After changing the regex, also adjust the time calculation below
    if your frame interval is not uniform.
    """
    match = re.search(r'_(\d{6})_1_debanded\.png$', filename)
    if match:
        frame_number = int(match.group(1))
        hours_per_segment = 24.0  # ⬅ CHANGE if your session spans a different duration (e.g. 12.0 for 12-hour sessions)
        time_increment = hours_per_segment / max_frames
        return start_hour + (frame_number - 1) * time_increment
    return None

def align_image_ecc(target, reference):
    """
    Aligns 'target' image to 'reference' using Enhanced Correlation Coefficient.
    Corrects for small camera repositioning or drift between frames.

    ⬇ OPTIONAL — Can be disabled if your camera is perfectly fixed:
    In the main processing loop below, replace:
        aligned = align_image_ecc(img, img_ref)
    with:
        aligned = img

    ⬇ TUNE IF NEEDED:
    - Max iterations (50): increase to 100–200 for higher precision, slower processing.
    - Epsilon (1e-5): decrease (e.g. 1e-7) for finer alignment convergence.
    """
    if len(reference.shape) == 3: reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if len(target.shape) == 3: target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(reference, target, warp_matrix, warp_mode, criteria)
        h, w = reference.shape
        aligned_image = cv2.warpAffine(target, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_image
    except Exception:
        return None

def get_indent_intensity(img, cx, cy, radius=5):
    """
    Measures mean pixel intensity in a circular region around (cx, cy).
    This is the core measurement — changes in intensity reflect changes
    in indent contrast, which correlates with fatigue-induced surface damage.

    ⬇ TUNE IF NEEDED:
    - radius=5: sampling circle radius in pixels.
      Increase (e.g. 10–15) for larger indents.
      Decrease (e.g. 2–3) for very small or tightly spaced indents.
      A good starting point is roughly half the visible indent diameter.
    """
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)
    cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
    mean_val = cv2.mean(img, mask=mask)[0]
    return mean_val

# ==========================================
# 4. MAIN PROCESSING
# ==========================================

if __name__ == "__main__":
    
    # Storage for all indent data across all sessions
    # Key: Indent ID (1-based integer), Value: {'times': [], 'depths': []}
    global_indent_data = {}
    
    verification_images = []
    
    all_files = os.listdir(SEARCH_DIR)

    # --- SET UP GLOBAL BRIGHTNESS REFERENCE ---
    # Takes the very first image of the first session as the brightness standard.
    # All subsequent images are linearly normalized to match it.
    first_prefix = SEGMENT_CONFIG[0][0]
    first_files = [f for f in all_files if f.startswith(first_prefix) and f.endswith(FILE_EXTENSION)]
    first_files.sort(key=natural_sort_key)
    
    if first_files:
        master_ref_path = os.path.join(SEARCH_DIR, first_files[0])
        master_ref_img = cv2.imread(master_ref_path)
        master_ref_gray = cv2.cvtColor(master_ref_img, cv2.COLOR_BGR2GRAY)
        master_ref_stats = cv2.meanStdDev(master_ref_gray)
        print("✓ Global Brightness Reference Established.")
    else:
        print("CRITICAL: No files found to establish reference.")
        sys.exit()

    prev_segment_ref = None
    prev_segment_targets = None

    # --- MAIN LOOP OVER SESSIONS ---
    for seg_idx, (prefix, start_hour, max_frames) in enumerate(SEGMENT_CONFIG):
        print(f"\n{'='*40}")
        print(f"PROCESSING BATCH: {prefix}")
        print(f"  Time Range: {start_hour}–{start_hour+24} hours")
        print(f"  Expected Frames: {max_frames}")
        print(f"{'='*40}")
        
        batch_files = [f for f in all_files if f.startswith(prefix) and f.endswith(FILE_EXTENSION)]
        batch_files.sort(key=natural_sort_key)
        
        if not batch_files:
            print(f"  ⚠ No files found for {prefix}")
            continue
        
        print(f"  Found {len(batch_files)} files")
            
        # Load and normalize the first image of this session as the alignment reference
        ref_path = os.path.join(SEARCH_DIR, batch_files[0])
        img_ref_raw = cv2.imread(ref_path)
        
        if img_ref_raw is None:
            print(f"  ⚠ Could not read reference image: {ref_path}")
            continue
        
        img_ref = normalize_brightness(img_ref_raw, master_ref_stats)
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        
        # Load or create the indent template
        template = load_or_create_template(img_ref_gray)
        if template is None:
            print("  ⚠ Template creation failed")
            sys.exit()
            
        # --- MANUAL INDENT SELECTION ---
        # An interactive window will open. Click each indent you want to track,
        # then press C to confirm. You can select any number of indents.
        targets = select_indents_manual_roi(img_ref, template, title_prefix=prefix)
        if not targets:
            print(f"  ⚠ No indents selected for {prefix}, skipping...")
            continue
        
        prev_segment_ref = img_ref.copy()
        prev_segment_targets = targets
            
        print(f"✓ Tracking {len(targets)} indents.")
        
        # Initialize tracking state for each selected indent
        active_indents = []
        for i, (tx, ty) in enumerate(targets):
            t_template = get_tracking_template(img_ref, tx, ty, size=40)  # ⬅ TUNE: size=40 is the patch size in pixels
            
            indent_id = i + 1
            if indent_id not in global_indent_data:
                global_indent_data[indent_id] = {'times': [], 'depths': []}
                
            active_indents.append({
                'id': indent_id,
                'current_x': tx,
                'current_y': ty,
                'template': t_template,
                'seg_times': [],
                'seg_depths': []
            })
        
        # --- PROCESS EACH FRAME IN THIS SESSION ---
        for i, fname in enumerate(batch_files):
            path = os.path.join(SEARCH_DIR, fname)
            img_raw = cv2.imread(path)
            if img_raw is None:
                print(f"  ⚠ Could not read: {fname}")
                continue
            
            img = normalize_brightness(img_raw, master_ref_stats)
            
            # Calculate time for this frame
            time_val = get_time_from_filename(fname, start_hour, max_frames)
            if time_val is None:
                time_val = start_hour + (i * (24.0 / max_frames))  # ⬅ CHANGE 24.0 if session duration differs
            
            # Align to the session reference image
            if i == 0:
                aligned = img
            else:
                aligned = align_image_ecc(img, img_ref)
                # ⬇ OPTIONAL: If you want to skip alignment entirely, comment the 2 lines above
                # and uncomment the line below:
                # aligned = img
                if aligned is None:
                    print(f"  ⚠ Alignment failed for frame {i}")
                    continue
            
            # Track and measure each indent
            for indent in active_indents:
                cx, cy, conf = track_template_local(aligned, indent['current_x'], indent['current_y'], indent['template'], search_window=80)
                
                # If confidence is low, retry with a wider search window
                if conf <= 0.6:
                    cx, cy, conf = track_template_local(aligned, indent['current_x'], indent['current_y'], indent['template'], search_window=120)
                    if conf <= 0.5:
                        # Fall back to last known position
                        cx, cy = indent['current_x'], indent['current_y']

                # Periodically refresh tracking template from current frame appearance
                if i % 5 == 0 and conf > 0.8:
                    indent['template'] = get_tracking_template(aligned, cx, cy, size=40)
                
                indent['current_x'] = cx
                indent['current_y'] = cy
                
                # ⬇ TUNE: radius=8 is the measurement circle size in pixels
                depth = get_indent_intensity(aligned, cx, cy, radius=8)
                
                indent['seg_times'].append(time_val)
                indent['seg_depths'].append(depth)
            
            # Save a verification snapshot every 25 frames
            # ⬇ OPTIONAL: Change the interval (25) or comment out this block entirely
            # if you don't need the verification grid output
            if i % 25 == 0 and len(verification_images) < VERIFY_SAMPLES:
                vis = aligned.copy()
                for indent in active_indents:
                    ix, iy = indent['current_x'], indent['current_y']
                    cv2.circle(vis, (ix, iy), 10, (0, 0, 255), 2)
                    cv2.putText(vis, f"{indent['id']}", (ix+15, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                verification_images.append(vis)
                
            if i % 10 == 0:
                print(f"\r  > Frame {i+1}/{len(batch_files)} processed", end="")
        
        print()  # New line after progress bar
        
        # --- STITCH SESSIONS TOGETHER ---
        # Offsets each new session's intensity to match where the previous session ended.
        # This removes artificial jumps caused by camera repositioning between days.
        # ⬇ OPTIONAL: Set STITCH_DATA = False at the top of the script to disable this.
        if STITCH_DATA:
            for indent in active_indents:
                gid = indent['id']
                g_depths = global_indent_data[gid]['depths']
                s_depths = indent['seg_depths']
                
                if g_depths and s_depths:
                    prev_end = np.median(g_depths[-5:])
                    curr_start = np.median(s_depths[:5])
                    offset = curr_start - prev_end
                    s_depths = [d - offset for d in s_depths]
                    print(f"  > Indent {gid}: Applied stitching offset: {offset:.2f}")
                
                global_indent_data[gid]['times'].extend(indent['seg_times'])
                global_indent_data[gid]['depths'].extend(s_depths)
        else:
            for indent in active_indents:
                gid = indent['id']
                global_indent_data[gid]['times'].extend(indent['seg_times'])
                global_indent_data[gid]['depths'].extend(indent['seg_depths'])

    # ==========================================
    # 5. PLOTTING & ANALYSIS
    # ==========================================

    if any(len(d['times']) > 10 for d in global_indent_data.values()):
        print("\n\nRunning Analysis...")
        
        num_indents = len(global_indent_data)
        colors = cm.rainbow(np.linspace(0, 1, num_indents))
        
        # ===== COMBINED PLOT — all indents overlaid on one chart =====
        # ⬇ OPTIONAL: Comment out this entire block if you only want individual plots
        fig, ax = plt.subplots(figsize=(16, 10))
        
        for idx, (gid, data) in enumerate(global_indent_data.items()):
            times = np.array(data['times'])
            depths = np.array(data['depths'])
            
            if len(times) < 10: continue
            
            # K-Means clustering splits measurements into 2 groups (e.g. early vs late stage)
            # ⬇ TUNE: n_clusters=2 — change to 3 if you expect 3 distinct fatigue stages
            X = np.column_stack((times, depths))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            mask0 = labels == 0
            mask1 = labels == 1
            
            ax.scatter(times[mask0], depths[mask0], 
                        color=colors[idx], marker='o', s=20, alpha=0.6, 
                        edgecolors='none')
            
            ax.scatter(times[mask1], depths[mask1], 
                        color=colors[idx], marker='x', s=30, alpha=0.8,
                        linewidths=1.5)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        shape_handles = [
            Line2D([0], [0], marker='o', color='w', label='Cluster A', 
                   markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='Cluster B', 
                   markeredgecolor='gray', markersize=8, markeredgewidth=2)
        ]
        
        indent_handles = []
        for idx, (gid, data) in enumerate(global_indent_data.items()):
            if len(data['times']) >= 10:
                indent_handles.append(Line2D([0], [0], color=colors[idx], lw=3, 
                                            label=f'Indent {gid}'))
        
        legend1 = ax.legend(handles=shape_handles, loc='lower left', 
                           title="Stages", fontsize=9, framealpha=0.9)
        ax.add_artist(legend1)
        
        ax.legend(handles=indent_handles, bbox_to_anchor=(1.02, 0.5), 
                 loc='center left', title="Indents", fontsize=8, 
                 framealpha=0.9, ncol=1)
            
        ax.set_xlabel("Time (Hours)", fontsize=12)
        ax.set_ylabel("Normalized Indent Intensity", fontsize=12)
        ax.set_title("Fatigue Stage Analysis (Multi-Indent Tracking)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("clustering_result_combined_1.png", dpi=150, bbox_inches='tight')
        plt.show()
        # ===== END COMBINED PLOT =====

        # ===== INDIVIDUAL PLOTS — one subplot per indent =====
        # ⬇ OPTIONAL: Comment out this entire block if you only want the combined plot
        print("\nGenerating individual indent plots...")
        
        n_valid_indents = sum(1 for data in global_indent_data.values() if len(data['times']) >= 10)
        cols = 5  # ⬅ CHANGE: columns in the subplot grid (e.g. 3 for ~9 indents, 6 for 30+ indents)
        rows = (n_valid_indents + cols - 1) // cols
        
        fig2, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        for idx, (gid, data) in enumerate(global_indent_data.items()):
            times = np.array(data['times'])
            depths = np.array(data['depths'])
            
            if len(times) < 10: continue
            
            ax = axes[plot_idx]
            
            X = np.column_stack((times, depths))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # ⬅ TUNE: n_clusters
            labels = kmeans.fit_predict(X_scaled)
            
            mask0 = labels == 0
            mask1 = labels == 1
            
            ax.scatter(times[mask0], depths[mask0], 
                      color=colors[idx], marker='o', s=25, alpha=0.7, 
                      label='Cluster A', edgecolors='none')
            
            ax.scatter(times[mask1], depths[mask1], 
                      color=colors[idx], marker='x', s=35, alpha=0.9,
                      label='Cluster B', linewidths=2)
            
            ax.set_xlabel("Time (Hours)", fontsize=10)
            ax.set_ylabel("Indent Intensity", fontsize=10)
            ax.set_title(f"Indent {gid}", fontsize=12, fontweight='bold', color=colors[idx])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            plot_idx += 1
        
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("clustering_result_individual_1.png", dpi=150, bbox_inches='tight')
        plt.show()
        # ===== END INDIVIDUAL PLOTS =====

        # ===== VERIFICATION GRID — visual check of tracking accuracy =====
        # ⬇ OPTIONAL: Comment out this entire block if you don't need visual verification
        if verification_images:
            plt.figure(figsize=(16, 12))
            num_imgs = len(verification_images)
            cols = 4  # ⬅ CHANGE: columns in verification grid layout
            rows = (num_imgs + cols - 1) // cols
            for k in range(num_imgs):
                plt.subplot(rows, cols, k+1)
                plt.imshow(cv2.cvtColor(verification_images[k], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title("Tracking Verify")
            plt.tight_layout()
            plt.savefig("verification_grid_1.png")
            plt.show()
        # ===== END VERIFICATION GRID =====

    else:
        print("\n⚠ Not enough data collected for analysis.")
        print("Please check that:")
        print("  1. Image files are in the correct directory (SEARCH_DIR)")
        print("  2. File names match the expected prefix and FILE_EXTENSION")
        print("  3. Indents were successfully selected and tracked")
