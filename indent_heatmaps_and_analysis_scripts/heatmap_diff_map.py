import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# ⬇ CHANGE THESE: Replace with your own image filenames.
# All four files must be in the same folder as this script.
# Format: one reference (0h) image + one per time point.
#
# To add more time points (e.g. 96h, 120h):
#   1. Add a new FILENAME_XXH variable below
#   2. Add OUT_ALIGNED_XXH and OUT_DIFF_XXH output names
#   3. In __main__, add load_image(), align_and_diff(), and a new plt.subplot()
FILENAME_REF = "TS-20170228220046873_INDENT01_4.5X_01_after_polish.tif"  # 0h reference — always keep this
FILENAME_24H = "TS-20170228220046873_INDENT01_4.5X_01_24hours.tif"
FILENAME_48H = "TS-20170228220046873_INDENT01_4.5X_01_48hours.tif"
FILENAME_72H = "TS-20170228220046873_INDENT01_4.5X_01_72hours.tif"

# ⬇ CHANGE THESE (optional): Output filenames for saved results.
# These are written to the same folder as the script.
OUT_ALIGNED_24 = "aligned_24h_1x1.tif"
OUT_ALIGNED_48 = "aligned_48h_1x1.tif"
OUT_ALIGNED_72 = "aligned_72h_1x1.tif"

OUT_DIFF_24 = "diff_map_24h_1x1.png"
OUT_DIFF_48 = "diff_map_48h_1x1.png"
OUT_DIFF_72 = "diff_map_72h_1x1.png"

def load_image(filename):
    """
    Loads image in its original bit-depth (8-bit, 16-bit, etc.).
    Works with .tif, .png, .jpg — no changes needed here.
    """
    print(f"Loading {filename}...")
    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        return None
    img = cv2.imread(filename, -1)
    if img is None:
        print(f"ERROR: Could not decode {filename}.")
        return None
    return img

def align_and_diff(target, reference, label):
    """
    Aligns 'target' image to 'reference' using ORB feature matching + Homography,
    then computes a gamma-enhanced absolute difference map.

    ⬇ TUNE IF NEEDED:
    - nfeatures=50000: number of ORB keypoints detected. Lower (e.g. 10000) is faster
      but less accurate. Increase if alignment fails on low-texture images.
    - good_matches top 15%: increase fraction (e.g. 0.25) if alignment is unstable,
      decrease (e.g. 0.10) to use only the very best matches.
    - RANSAC threshold 5.0: increase (e.g. 10.0) if images have large distortion,
      decrease (e.g. 2.0) for very precise alignment.
    - gamma=0.6: controls contrast enhancement of the diff map.
      Lower value (e.g. 0.3) makes faint changes more visible,
      higher value (e.g. 1.0) shows only strong changes.
    """
    print(f"--- Processing {label} ---")

    # 1. Get dimensions
    h, w = reference.shape[:2]

    # 2. Normalize to 8-bit grayscale for feature detection
    ref_8bit = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    tgt_8bit = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    if len(ref_8bit.shape) == 3:
        ref_8bit = cv2.cvtColor(ref_8bit, cv2.COLOR_BGR2GRAY)
    if len(tgt_8bit.shape) == 3:
        tgt_8bit = cv2.cvtColor(tgt_8bit, cv2.COLOR_BGR2GRAY)

    # 3. Detect ORB features
    orb = cv2.ORB_create(nfeatures=50000)  # ⬅ TUNE: reduce for speed, increase for accuracy
    kp1, des1 = orb.detectAndCompute(tgt_8bit, None)
    kp2, des2 = orb.detectAndCompute(ref_8bit, None)

    if des1 is None or des2 is None:
        print(f"WARNING: No features found in {label}.")
        return None, None

    # 4. Match features and keep top 15%
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:int(len(matches) * 0.15)]  # ⬅ TUNE: fraction of matches to use
    print(f"  Matches found: {len(matches)} | Used: {len(good_matches)}")

    if len(good_matches) < 10:
        print("  CRITICAL ERROR: Not enough matches to align.")
        return None, None

    # 5. Compute homography and warp
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # ⬅ TUNE: RANSAC threshold
    aligned_target = cv2.warpPerspective(target, M, (w, h))

    # 6. Compute absolute difference
    ref_float = reference.astype(np.float32)
    aligned_float = aligned_target.astype(np.float32)
    diff = cv2.absdiff(ref_float, aligned_float)

    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 7. Normalize and apply gamma contrast enhancement
    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gamma = 0.6  # ⬅ TUNE: lower = amplify faint changes, higher = show only strong changes
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    diff_enhanced = cv2.LUT(diff_vis, lookUpTable)

    return aligned_target, diff_enhanced

# =============================================================================
# --- MAIN EXECUTION ---
# =============================================================================

if __name__ == "__main__":
    print("Starting High-Res Titanium Analysis (0h vs 24h vs 48h vs 72h)...")

    # Load all images
    # ⬇ If adding more time points, add load_image() calls here
    img_0h  = load_image(FILENAME_REF)
    img_24h = load_image(FILENAME_24H)
    img_48h = load_image(FILENAME_48H)
    img_72h = load_image(FILENAME_72H)

    images = [img_0h, img_24h, img_48h, img_72h]
    if all(img is not None for img in images):

        # Process each time point against the 0h reference
        # ⬇ To add more time points, copy one of these blocks and update variables
        aligned_24, diff_24 = align_and_diff(img_24h, img_0h, "24 Hours")
        if aligned_24 is not None:
            cv2.imwrite(OUT_ALIGNED_24, aligned_24)
            cv2.imwrite(OUT_DIFF_24, diff_24)

        aligned_48, diff_48 = align_and_diff(img_48h, img_0h, "48 Hours")
        if aligned_48 is not None:
            cv2.imwrite(OUT_ALIGNED_48, aligned_48)
            cv2.imwrite(OUT_DIFF_48, diff_48)

        aligned_72, diff_72 = align_and_diff(img_72h, img_0h, "72 Hours")
        if aligned_72 is not None:
            cv2.imwrite(OUT_ALIGNED_72, aligned_72)
            cv2.imwrite(OUT_DIFF_72, diff_72)

        # --- Visualization ---
        # ⬇ CHANGE: Update subplot count (1, 3, ...) if you add more time points
        # ⬇ CHANGE: cmap='inferno' — alternatives: 'hot', 'plasma', 'jet', 'gray'
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        if diff_24 is not None:
            plt.imshow(diff_24, cmap='inferno')
            plt.title("Change: 0h vs 24h")
            plt.axis('off')

        plt.subplot(1, 3, 2)
        if diff_48 is not None:
            plt.imshow(diff_48, cmap='inferno')
            plt.title("Change: 0h vs 48h")
            plt.axis('off')

        plt.subplot(1, 3, 3)
        if diff_72 is not None:
            plt.imshow(diff_72, cmap='inferno')
            plt.title("Change: 0h vs 72h")
            plt.axis('off')

        plt.tight_layout()
        print("Analysis Complete. Images saved.")
        plt.show()
    else:
        print("Process stopped: One or more images failed to load.")
