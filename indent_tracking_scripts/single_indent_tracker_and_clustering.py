import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# ⬇ CHANGE THIS: One entry per experimental session/day.
# Format: ("image_filename_prefix", start_hour)
# Add or remove lines to match your number of sessions.
SEGMENT_CONFIG = [
    ("Nov_17", 0),   # Day 1: 0 to 24 hrs
    ("Nov_19", 24),  # Day 2: 24 to 48 hrs
    ("Nov_20", 48)   # Day 3: 48 to 72 hrs
]

# ⬇ CHANGE THIS: Extension of your image files. Common: ".png", ".jpg", ".tif"
FILE_EXTENSION = ".png"

# ⬇ CHANGE THIS: Folder containing your images.
# "." means same folder as this script.
# For a specific path use e.g.: r"C:\Users\YourName\experiment_images"
SEARCH_DIR = "."

# ⬇ CHANGE THIS (optional): Name of saved template file.
# Delete this file to force re-selection of a new template on next run.
TEMPLATE_FILENAME = "indent_template.png"

# --- SETTINGS ---
ALIGN_METHOD = "ECC"     # Alignment method — keep as "ECC" for best accuracy
REMOVE_ZEROS = True      # Filter frames where intensity drops to zero (camera dropout)
STITCH_DATA = True       # Smooth intensity jumps between sessions — recommended ON
VERIFY_SAMPLES = 15      # Max verification snapshots saved — increase for more checkpoints

# Global variables for mouse interaction — do not change
selected_point = None
confirmed_indent = None

# ==========================================
# 0. BRIGHTNESS NORMALIZATION
# ==========================================

def normalize_brightness(img, ref_stats):
    """
    Adjusts the brightness/contrast of 'img' to match the mean/std-dev 
    of the master reference image.
    Corrects for lighting drift between sessions — do not remove.
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
    Loads template from file or opens a selection window if not found.
    The template is a small patch around one indent — used to locate others.
    """
    if os.path.exists(TEMPLATE_FILENAME):
        template = cv2.imread(TEMPLATE_FILENAME, 0)
        
        if template is None:
            print(f"⚠ Found {TEMPLATE_FILENAME} but could not read it. Please re-select.")
        else:
            if template.shape[0] > img_gray.shape[0] // 2:
                template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
            return template
    
    print("\n⚠ 'indent_template.png' not found or invalid!")
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
    - threshold: lower (e.g. 0.25) finds more candidates, higher (e.g. 0.5) is stricter.
    - min_dist: increase if the same indent is detected twice, decrease if nearby indents are merged.
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
    Extracts a patch around (x, y) to use as the tracking template.

    ⬇ TUNE IF NEEDED:
    - size=40: patch size in pixels. Increase (e.g. 60–80) for larger indents,
      decrease (e.g. 20) for very small indents.
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

    ⬇ TUNE IF NEEDED:
    - search_window=80: search radius in pixels.
      Increase (e.g. 150) if the specimen moves a lot between frames.
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

def mouse_callback(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)

def select_indent_from_grid(img_color, candidates, title_prefix=""):
    """
    Opens an interactive window showing all detected indent candidates.
    Click near the ONE indent you want to track, then press C to confirm.

    Controls:
      - LEFT CLICK near the indent you want to track
      - Press 'C' to confirm selection
      - Press ESC to cancel (this session will be skipped)

    ℹ This script tracks ONE indent at a time.
    If you need to track multiple indents simultaneously, use
    cantilever_tracker_v1.py or cantilever_tracker_v2.py instead.
    """
    global selected_point
    selected_point = None
    
    print("\n" + "="*60)
    print(f"INDENT SELECTION: {title_prefix}")
    print("="*60)
    print("CLICK near the specific indent. Press 'C' to confirm.")
    print("="*60 + "\n")
    
    window_name = f"Select Indent: {title_prefix}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = img_color.shape[:2]
    scale_factor = 1.0
    if w > 1200:
        scale_factor = 1200 / w
        
    final_selection = None

    while True:
        display_img = img_color.copy()
        for (cx, cy) in candidates:
            cv2.circle(display_img, (cx, cy), 15, (0, 255, 0), 1)
            cv2.circle(display_img, (cx, cy), 2, (0, 255, 0), -1)

        if selected_point:
            mx, my = selected_point
            nearest_dist = float('inf')
            nearest_cand = None
            
            for (cx, cy) in candidates:
                dist = np.sqrt((cx - mx)**2 + (cy - my)**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_cand = (cx, cy)
            
            if nearest_cand and nearest_dist < 100:  # ⬅ TUNE: snap radius in pixels (100 = snaps if click is within 100px of a candidate)
                start_x, start_y = nearest_cand
            else:
                start_x, start_y = mx, my
                
            final_selection = (start_x, start_y)
            cv2.circle(display_img, (start_x, start_y), 20, (0, 0, 255), 3)
            cv2.circle(display_img, (start_x, start_y), 2, (0, 0, 255), -1)
            cv2.putText(display_img, "TARGET", (start_x+25, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if scale_factor != 1.0:
            disp_h, disp_w = int(h*scale_factor), int(w*scale_factor)
            view_img = cv2.resize(display_img, (disp_w, disp_h))
        else:
            view_img = display_img
            
        cv2.imshow(window_name, view_img)
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord('c') or key == ord('C'):
            if final_selection:
                cv2.destroyWindow(window_name)
                return final_selection
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None

# ==========================================
# 3. ANALYSIS UTILS
# ==========================================

def natural_sort_key(s):
    """Sorts filenames naturally (e.g. frame2 before frame10). Do not change."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_time_from_filename(filename, start_hour):
    """
    Extracts frame number from filenames in the format: 'Prefix (N).png'
    and converts it to a time value in hours.

    ⬇ CHANGE THIS if your filenames use a different format.
    For example, if files are named 'frame_001.png':
        match = re.search(r'frame_(\d+)', filename)
        if match:
            frame_number = int(match.group(1))
            return start_hour + (frame_number - 1) * 0.25  # adjust 0.25 to your frame interval in hours
    """
    match = re.search(r'\s\((\d+)\)', filename)
    if match:
        frame_number = int(match.group(1))
        return start_hour + (frame_number - 1) * 0.25  # ⬅ CHANGE 0.25 to your actual time interval between frames (hours)
    return None

def align_image_ecc(target, reference):
    """
    Aligns 'target' to 'reference' using Enhanced Correlation Coefficient.
    Corrects for small camera movements between frames.

    ⬇ OPTIONAL — disable if camera is perfectly fixed. In the main loop,
    replace the align_image_ecc call with: aligned = img

    ⬇ TUNE IF NEEDED:
    - Max iterations (50): increase to 100–200 for higher precision, slower speed.
    - Epsilon (1e-5): decrease (e.g. 1e-7) for finer convergence.
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
    Changes in this value over time reflect indent contrast changes from fatigue.

    ⬇ TUNE IF NEEDED:
    - radius=5: sampling circle radius in pixels.
      Increase (e.g. 10–15) for larger indents, decrease (e.g. 2–3) for small ones.
      Good starting point: roughly half the visible indent diameter.
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
    
    final_times = []
    final_depths = []
    verification_images = []
    
    all_files = os.listdir(SEARCH_DIR)

    # --- SET UP GLOBAL BRIGHTNESS REFERENCE ---
    # Uses the very first image of the first session as the brightness standard.
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

    # --- MAIN LOOP OVER SESSIONS ---
    for seg_idx, (prefix, start_hour) in enumerate(SEGMENT_CONFIG):
        print(f"\n{'='*40}")
        print(f"PROCESSING BATCH: {prefix}")
        print(f"{'='*40}")
        
        batch_files = [f for f in all_files if f.startswith(prefix) and f.endswith(FILE_EXTENSION)]
        batch_files.sort(key=natural_sort_key)
        
        if not batch_files: continue
            
        # Load and normalize the first image of this session
        ref_path = os.path.join(SEARCH_DIR, batch_files[0])
        img_ref_raw = cv2.imread(ref_path)
        img_ref = normalize_brightness(img_ref_raw, master_ref_stats)
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        
        # Load or create the indent template
        template = load_or_create_template(img_ref_gray)
        if template is None: sys.exit()
            
        candidates = detect_grid_candidates(img_ref_gray, template)
        
        # --- MANUAL SELECTION ---
        # A window opens showing all detected candidates (green circles).
        # Click the ONE indent you want to track, then press C.
        target = select_indent_from_grid(img_ref, candidates, title_prefix=prefix)
        if target is None: continue
            
        target_x, target_y = target
        print(f"✓ Tracking: ({target_x}, {target_y})")
        
        tracking_template = get_tracking_template(img_ref, target_x, target_y, size=40)  # ⬅ TUNE: size=40 is patch size in pixels
        
        current_times = []
        current_depths = []
        
        # --- PROCESS EACH FRAME ---
        for i, fname in enumerate(batch_files):
            path = os.path.join(SEARCH_DIR, fname)
            img_raw = cv2.imread(path)
            if img_raw is None: continue
            
            img = normalize_brightness(img_raw, master_ref_stats)
            
            time_val = get_time_from_filename(fname, start_hour)
            if time_val is None: time_val = start_hour + (i * 0.25)  # ⬅ CHANGE 0.25 to your frame interval (hours)
            
            # Align to session reference
            if i == 0:
                aligned = img
            else:
                aligned = align_image_ecc(img, img_ref)
                # ⬇ OPTIONAL: To skip alignment entirely, comment the 2 lines above
                # and uncomment the line below:
                # aligned = img
                if aligned is None: continue
            
            # Track the indent position in this frame
            current_x, current_y, conf = track_template_local(aligned, target_x, target_y, tracking_template, search_window=80)
            
            # If confidence is low, retry with wider search
            if conf <= 0.6:
                current_x, current_y, conf = track_template_local(aligned, target_x, target_y, tracking_template, search_window=120)
                if conf <= 0.5:
                    # Fall back to last known position
                    current_x, current_y = target_x, target_y

            # Periodically refresh tracking template from current frame
            if i % 5 == 0 and conf > 0.8:
                tracking_template = get_tracking_template(aligned, current_x, current_y, size=40)

            # Measure indent intensity
            depth = get_indent_intensity(aligned, current_x, current_y, radius=8)  # ⬅ TUNE: radius=8 in pixels
            
            current_times.append(time_val)
            current_depths.append(depth)
            
            # Save a verification snapshot every 25 frames
            # ⬇ OPTIONAL: Change interval (25) or comment out this block if not needed
            if i % 25 == 0 and len(verification_images) < VERIFY_SAMPLES:
                vis = aligned.copy()
                cv2.circle(vis, (current_x, current_y), 10, (0, 0, 255), 2)
                cv2.putText(vis, f"{time_val:.1f}h", (current_x+15, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                verification_images.append(vis)
                
            if i % 10 == 0:
                print(f"\r  > Frame {i}/{len(batch_files)} processed", end="")
        
        # --- STITCH SESSIONS TOGETHER ---
        # Offsets current session to match where the previous session ended.
        # ⬇ OPTIONAL: Set STITCH_DATA = False at the top to disable.
        if STITCH_DATA and final_depths and current_depths:
            prev_end = np.median(final_depths[-5:])
            curr_start = np.median(current_depths[:5])
            offset = curr_start - prev_end
            current_depths = [d - offset for d in current_depths]
            print(f"\n  > Applied stitching offset: {offset:.2f}")
            
        final_times.extend(current_times)
        final_depths.extend(current_depths)

    # ==========================================
    # 5. PLOTTING & ANALYSIS
    # ==========================================

    if len(final_times) > 10:
        print("\n\nRunning Analysis...")
        
        # K-Means clustering: splits data into 2 groups (early vs late fatigue stage)
        # ⬇ TUNE: n_clusters=2 — change to 3 if you expect 3 distinct fatigue stages
        X = np.column_stack((final_times, final_depths))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # ===== MAIN PLOT =====
        # ⬇ OPTIONAL: Comment out this block if you don't need the clustering plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(final_times, final_depths, c=labels, cmap='viridis', s=20, alpha=0.7)
        plt.xlabel("Time (Hours)")
        plt.ylabel("Normalized Indent Intensity")
        plt.title("Fatigue Stage Clustering (Brightness Normalized)")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("clustering_result.png")
        plt.show()
        # ===== END MAIN PLOT =====

        # ===== VERIFICATION GRID =====
        # ⬇ OPTIONAL: Comment out this block if you don't need visual tracking verification
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
            plt.savefig("verification_grid.png")
            plt.show()
        # ===== END VERIFICATION GRID =====
    else:
        print("Not enough data.")
