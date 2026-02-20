import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# ⬇ CHANGE THIS: Full path to the folder containing your images and template.
# Windows example: r"C:\Users\YourName\experiment_images"
# Mac/Linux example: "/home/yourname/experiment_images"
IMAGE_DIR = r"c:/Users/harsh/OneDrive/Desktop/Coding/Python code/Cantilever Experiment readings/Cantilever Exp Image Indent Analysis"

# ⬇ CHANGE THIS: List your image filenames in chronological order.
# First entry = reference (0h/after polish), then each subsequent time point.
# Add or remove entries to match your experiment (e.g. add "..._96hour.tif" for a 4th point).
IMAGE_FILES = [
    "TS-20170228203451635_1X_01_after_polish.tif",    # 0h reference
    "TS-20170228203451635_1X_01_After_24hour.tif",    # 24h
    "TS-20170228203451635_1X_01_After_48hour.tif",    # 48h
    "TS-20170228203451635_1X_01_After_72hour.tif"     # 72h
]

# ⬇ CHANGE THIS: Set to match your indent size in pixels.
# INDENT_RADIUS: should roughly cover the full indent (increase for larger indents)
# GRID_SIZE: total number of indents in your grid (e.g. 16 for 4x4, 25 for 5x5, 36 for 6x6)
INDENT_RADIUS = 50   # ⬅ pixels — increase for larger indents, decrease for smaller
GRID_SIZE = 25       # ⬅ expected number of indents in the grid (used to cap detections)

def load_image(filename):
    """
    Loads a grayscale image from IMAGE_DIR.
    Supports .tif, .png, .jpg — no changes needed here.
    """
    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        if os.path.exists(filename):
            full_path = filename
        else:
            print(f"Error: File not found at {full_path}")
            return None
    img = cv2.imread(full_path, 0)
    return img

def load_template():
    """
    Loads 'indent_template.png' from IMAGE_DIR.

    ⬇ CHANGE THIS if your template file has a different name:
        template_path = os.path.join(IMAGE_DIR, "your_template_name.png")

    The template should be a tight crop around ONE representative indent.
    If you don't have one yet, crop it manually from your reference image
    and save it as 'indent_template.png' in your IMAGE_DIR.

    ⬇ TUNE IF NEEDED:
    - crop_size=40: keeps the central 40x40 pixels of the template.
      Increase (e.g. 60–80) if your indents are large,
      decrease (e.g. 20–30) if they are very small or tightly packed.
    """
    template_path = os.path.join(IMAGE_DIR, "indent_template.png")
    if not os.path.exists(template_path):
        if os.path.exists("indent_template.png"):
            template_path = "indent_template.png"
        else:
            print(f"Error: Template file not found at {template_path}")
            return None

    template = cv2.imread(template_path, 0)
    if template is None:
        print("Error loading template image.")
        return None

    # Slight blur to match image noise profile
    template = cv2.GaussianBlur(template, (5, 5), 0)

    # Crop to centre to remove excess background around the indent
    h, w = template.shape
    center_x, center_y = w // 2, h // 2
    crop_size = 40  # ⬅ TUNE: central crop size in pixels

    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w, center_x + crop_size // 2)
    y2 = min(h, center_y + crop_size // 2)

    template = template[y1:y2, x1:x2]
    return template

def get_indent_centers_template(img):
    """
    Finds all indent centres in 'img' using multi-scale template matching.

    ⬇ TUNE IF NEEDED:
    - threshold=0.35: detection sensitivity. Lower (e.g. 0.25) finds more candidates
      but may include false positives. Raise (e.g. 0.45) to be stricter.
    - min_distance=40: minimum pixel gap between two detections. Increase if the
      same indent is detected twice, decrease if adjacent indents are being merged.
    - fallback threshold=0.28: used when fewer than GRID_SIZE indents are found.
      Lower this further (e.g. 0.20) if indents are still being missed.
    - Row tolerance 60px: in grid sorting, points within 60px vertical distance
      are grouped into the same row. Adjust if your grid rows are closer or further apart.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    template = load_template()
    if template is None:
        print("CRITICAL ERROR: Could not load template. Aborting.")
        return [], np.zeros((10, 10))

    h, w = template.shape

    # Multi-scale matching at 90%, 100%, 110% of template size
    scales = [0.9, 1.0, 1.1]
    all_matches = []

    for scale in scales:
        if scale != 1.0:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            scaled_template = cv2.resize(template, (scaled_w, scaled_h))
        else:
            scaled_template = template
            scaled_h, scaled_w = h, w

        res = cv2.matchTemplate(blurred, scaled_template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.35  # ⬅ TUNE: lower = more detections, higher = stricter
        locations = np.where(res >= threshold)

        for pt in zip(*locations[::-1]):
            center_x = pt[0] + scaled_w // 2
            center_y = pt[1] + scaled_h // 2
            match_score = res[pt[1], pt[0]]
            all_matches.append((center_x, center_y, match_score, scale))

    print(f"  Total candidates found: {len(all_matches)}")

    # Non-maximum suppression — keep best match per region
    all_matches.sort(key=lambda x: x[2], reverse=True)

    detected_centers = []
    min_distance = 40  # ⬅ TUNE: min pixel distance between accepted detections

    for cx, cy, score, scale in all_matches:
        is_duplicate = False
        for existing_x, existing_y in detected_centers:
            distance = np.sqrt((cx - existing_x)**2 + (cy - existing_y)**2)
            if distance < min_distance:
                is_duplicate = True
                break

        if not is_duplicate:
            detected_centers.append((cx, cy))
            if len(detected_centers) >= GRID_SIZE:
                break

    print(f"  After filtering: {len(detected_centers)} indents")

    # Grid validation — remove stray outliers not part of a row structure
    if len(detected_centers) >= 15:
        temp_sorted = sorted(detected_centers, key=lambda p: p[1])
        rows = []
        current_row = []
        last_y = temp_sorted[0][1]

        for p in temp_sorted:
            if abs(p[1] - last_y) < 60:  # ⬅ TUNE: row grouping tolerance in pixels
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
                last_y = p[1]
        rows.append(current_row)

        valid_points = []
        for row in rows:
            if len(row) >= 3:  # ⬅ TUNE: minimum points per row to be considered valid
                valid_points.extend(row)

        if len(valid_points) > len(detected_centers) * 0.7:
            detected_centers = valid_points
            print(f"  After grid validation: {len(detected_centers)} indents")

    # Fallback: lower threshold if not enough indents found
    if len(detected_centers) < GRID_SIZE:
        print(f"  Only found {len(detected_centers)} indents. Trying fallback...")

        res = cv2.matchTemplate(blurred, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.28  # ⬅ TUNE: fallback threshold — lower if indents still being missed

        mask = np.ones_like(res, dtype=np.uint8)
        for ex, ey in detected_centers:
            cv2.circle(mask, (ex - w//2, ey - h//2), min_distance, 0, -1)

        res_masked = res * mask
        fallback_matches = []
        res_copy = res_masked.copy()

        needed = GRID_SIZE - len(detected_centers)
        for i in range(needed + 2):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_copy)
            if max_val < threshold:
                break
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            fallback_matches.append((center_x, center_y))
            cv2.circle(res_copy, max_loc, 30, -1, -1)

        detected_centers.extend(fallback_matches[:needed])
        print(f"  After fallback: {len(detected_centers)} indents")

    # Sort into grid order (row by row, left to right)
    detected_centers.sort(key=lambda p: p[1])
    rows = []
    current_row = []

    if detected_centers:
        last_y = detected_centers[0][1]
        for p in detected_centers:
            if abs(p[1] - last_y) < 60:  # ⬅ TUNE: same row tolerance as above
                current_row.append(p)
            else:
                rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [p]
                last_y = p[1]
        rows.append(sorted(current_row, key=lambda x: x[0]))

    sorted_centers = [p for row in rows for p in row]
    sorted_centers = sorted_centers[:GRID_SIZE]

    return sorted_centers, template

def get_weber_contrast(img, center_x, center_y, radius):
    """
    Calculates Weber Contrast at an indent location:
        Weber Contrast = (Background Mean - Indent Mean) / Background Mean

    A higher value means the indent is darker relative to its surroundings,
    indicating more visible surface damage.

    ⬇ TUNE IF NEEDED:
    - radius: passed in from INDENT_RADIUS — set at top of script.
    - Background ring width is fixed at +20 pixels beyond the indent radius.
      Change the '+ 20' below if you want a wider or narrower background ring.
    """
    mask_indent = np.zeros_like(img)
    cv2.circle(mask_indent, (int(center_x), int(center_y)), radius, 255, -1)

    mask_bg = np.zeros_like(img)
    cv2.circle(mask_bg, (int(center_x), int(center_y)), radius + 20, 255, -1)  # ⬅ TUNE: background ring width
    mask_bg = cv2.subtract(mask_bg, mask_indent)

    mean_indent = cv2.mean(img, mask=mask_indent)[0]
    mean_bg = cv2.mean(img, mask=mask_bg)[0]

    if mean_bg == 0: return 0
    return (mean_bg - mean_indent) / mean_bg

# =============================================================================
# --- MAIN EXECUTION LOOP ---
# =============================================================================

if __name__ == "__main__":

    all_results = {}

    for filename in IMAGE_FILES:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

        ref_img = load_image(filename)
        if ref_img is None:
            continue

        indent_centers, template = get_indent_centers_template(ref_img)
        print(f"Final count: {len(indent_centers)} indents.")

        base_name = os.path.splitext(filename)[0]

        # --- Verification plot: template used + detections overlaid ---
        # ⬇ OPTIONAL: Comment out this block if you don't need per-image verification plots
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(template, cmap='gray')
        ax[0].set_title("Template Used")
        ax[0].axis('off')

        debug_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
        for i, (x, y) in enumerate(indent_centers):
            label = f"{i+1}"
            cv2.putText(debug_img, label, (x-15, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.circle(debug_img, (x, y), INDENT_RADIUS, (0, 255, 0), 3)
            cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)

        ax[1].imshow(debug_img)
        ax[1].set_title(f"Detection: {filename} ({len(indent_centers)} found)")
        ax[1].axis('off')

        plt.tight_layout()
        verify_filename = f"verify_{base_name}.png"
        plt.savefig(verify_filename)
        print(f"Saved: {verify_filename}")
        plt.show()
        # ===== END VERIFICATION PLOT =====

        # Calculate Weber contrast for each indent and store
        if len(indent_centers) > 0:
            contrasts = []
            for (cx, cy) in indent_centers:
                val = get_weber_contrast(ref_img, cx, cy, INDENT_RADIUS)
                contrasts.append(val)
            all_results[filename] = contrasts
        else:
            print("No indents found to calculate contrast.")

    # --- Combined contrast plot across all time points ---
    # ⬇ OPTIONAL: Comment out this block if you only need the per-image verification plots
    print("\nGenerating combined contrast plot...")
    plt.figure(figsize=(14, 8))

    styles = ['-o', '-s', '-^', '-D']  # ⬅ CHANGE: line/marker styles per time point

    for i, filename in enumerate(IMAGE_FILES):
        if filename in all_results:
            data = all_results[filename]
            x_indices = np.arange(1, len(data) + 1)

            # ⬇ CHANGE: Update these label conditions to match your filename conventions
            if "polish" in filename.lower():
                label = "After Polish (0h)"
            elif "24hour" in filename.lower():
                label = "24 Hours"
            elif "48hour" in filename.lower():
                label = "48 Hours"
            elif "72hour" in filename.lower():
                label = "72 Hours"
            else:
                label = filename  # Falls back to filename if no keyword matched

            style = styles[i % len(styles)]
            plt.plot(x_indices, data, style, label=label, linewidth=2, markersize=6)

    plt.xlabel("Indent Number (1-25)")  # ⬅ CHANGE: update x-axis label if GRID_SIZE differs
    plt.ylabel("Weber Contrast")
    plt.title("Contrast Evolution of Indents Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    combined_filename = "combined_contrast_plot.png"
    plt.savefig(combined_filename)
    print(f"Saved: {combined_filename}")
    plt.show()
    # ===== END COMBINED PLOT =====

    print("\nBatch processing complete.")
