"""
L515 IR + Confidence + Depth
- Depth-gated template tracking (press 'c' to lock ROI on rockwool)
- EASY logging workflow for “wet vs dry separability up to X cm”

KEYS (main workflow)
  c : lock (capture template from current ROI) / start tracking
  r : reset / unlock
  1/2/3 : set distance label (default: 0.50m / 0.60m / 0.70m)
  w/d : set condition label to wet/dry
  g : "grab" a TRIAL (records N frames automatically, then writes 1 row to SUMMARY csv
      AND also writes per-frame rows to FRAMES csv)
  s : snapshot (saves arrays + preview image)
  e : toggle contrast (visual only)
  q : quit

What gets saved:
  - frames CSV: one row per frame during each trial
  - summary CSV: one row per trial (means/stds across frames)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
W, H, FPS = 320, 240, 30

ROI_R = 6                  # ROI half-size in ORIGINAL pixels (13x13)
SEARCH_R_LOCKED = 50       # search radius around last position while locked (original pixels)
MATCH_THRESH = 0.45        # template match score threshold (0..1)

DEPTH_JUMP_MAX = 0.20      # meters: reject tracker jumps beyond this from expected depth
DEPTH_ALPHA = 0.15         # smoothing factor for expected depth updates

# Logging distances (editable later)
DISTANCE_LABELS = ["0.50", "0.60", "0.70"]  # meters as strings (labels)
DISTANCE_KEYS = [ord('1'), ord('2'), ord('3')]

# Trial capture
TRIAL_FRAMES = 120         # ~4 seconds @ 30 fps (adjust as you like)
MIN_VALID_PCT = 60.0       # if a frame's depth valid% is below this, we still log it (flag), but note it.

# Display
WINDOW_NAME = "L515 IR (tracking + one-key trials)"
SCALE = 3                  # 2 or 3 recommended
BANNER_H = 150             # pixels (in DISPLAY space)

# Output
OUT_DIR = "runs"
SNAP_DIR = os.path.join(OUT_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
frames_csv_path  = os.path.join(OUT_DIR, f"{run_id}_frames.csv")
summary_csv_path = os.path.join(OUT_DIR, f"{run_id}_summary.csv")

# ----------------------------
# REALSENSE PIPELINE
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
config.enable_stream(rs.stream.confidence, W, H, rs.format.raw8, FPS)
config.enable_stream(rs.stream.infrared, W, H, rs.format.y8, FPS)

print("Starting...")
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# ----------------------------
# HELPERS
# ----------------------------
def clamp_box(cx, cy, r, w, h):
    x0 = max(int(cx - r), 0)
    x1 = min(int(cx + r + 1), w)
    y0 = max(int(cy - r), 0)
    y1 = min(int(cy + r + 1), h)
    return x0, x1, y0, y1

def roi_depth_median(depth_raw_u16, cx, cy, r, depth_scale):
    """Return (depth_med_m, valid_pct). Ignores zero-depth pixels."""
    x0, x1, y0, y1 = clamp_box(cx, cy, r, depth_raw_u16.shape[1], depth_raw_u16.shape[0])
    roi_raw = depth_raw_u16[y0:y1, x0:x1].astype(np.float32)
    roi_m = roi_raw * depth_scale
    valid = roi_m[roi_m > 0]
    if valid.size == 0:
        return 0.0, 0.0
    depth_med = float(np.median(valid))
    valid_pct = 100.0 * (valid.size / roi_m.size)
    return depth_med, valid_pct

def roi_stats(img_u8, cx, cy, r):
    """Return mean, std and ROI bounds for a uint8 image ROI."""
    x0, x1, y0, y1 = clamp_box(cx, cy, r, img_u8.shape[1], img_u8.shape[0])
    roi = img_u8[y0:y1, x0:x1].astype(np.float32)
    return float(roi.mean()), float(roi.std()), (x0, x1, y0, y1)

def ensure_csv_headers():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Frames CSV: one row per frame during a trial
    if not os.path.exists(frames_csv_path):
        with open(frames_csv_path, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow([
                "run_id",
                "trial_id",
                "timestamp",
                "condition",          # wet/dry/unknown
                "distance_label_m",   # label (0.50/0.60/0.70)
                "depth_med_m",
                "depth_valid_pct",
                "ir_mean",
                "ir_std",
                "conf_mean",
                "match_score",
                "expected_depth_m",
                "roi_cx",
                "roi_cy",
                "roi_r",
                "note"                # e.g., "low_valid"
            ])

    # Summary CSV: one row per trial
    if not os.path.exists(summary_csv_path):
        with open(summary_csv_path, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow([
                "run_id",
                "trial_id",
                "timestamp_start",
                "timestamp_end",
                "condition",
                "distance_label_m",
                "n_frames",
                "depth_med_mean",
                "depth_med_std",
                "depth_valid_mean",
                "ir_mean_mean",
                "ir_mean_std",
                "ir_std_mean",
                "conf_mean_mean",
                "match_mean",
                "notes"
            ])

def append_frames_rows(rows):
    with open(frames_csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)

def append_summary_row(row):
    with open(summary_csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ----------------------------
# MAIN
# ----------------------------
ensure_csv_headers()

try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W * SCALE + 40, H * SCALE + BANNER_H + 80)

    # Warm-up
    for _ in range(30):
        pipeline.wait_for_frames()

    # State
    locked = False
    roi_cx, roi_cy = W // 2, H // 2
    template = None
    expected_depth = None
    use_equalize = False

    # Labels
    condition = "unknown"         # set with w/d
    distance_label = DISTANCE_LABELS[0]  # set with 1/2/3

    # Trial capture state
    trial_active = False
    trial_id = 0
    trial_frames_left = 0
    trial_start_ts = None

    # Buffers for one trial
    trial_frame_rows = []   # rows for frames csv
    trial_stats = {
        "depth_med": [],
        "depth_valid": [],
        "ir_mean": [],
        "ir_std": [],
        "conf_mean": [],
        "match": []
    }

    while True:
        frames = pipeline.wait_for_frames()

        depth = frames.get_depth_frame()
        conf = None
        ir = None

        for f in frames:
            st = f.get_profile().stream_type()
            if st == rs.stream.confidence:
                conf = f
            elif st == rs.stream.infrared:
                ir = f

        if not depth or conf is None or ir is None:
            continue

        depth_raw = np.asanyarray(depth.get_data()).astype(np.uint16)
        conf_np = np.asanyarray(conf.get_data()).astype(np.uint8)
        ir_np = np.asanyarray(ir.get_data()).astype(np.uint8)

        # ----------------------------
        # TRACKING: template match + depth gate
        # ----------------------------
        match_score = None
        candidate_cx, candidate_cy = roi_cx, roi_cy

        if locked and template is not None:
            sx0, sx1, sy0, sy1 = clamp_box(roi_cx, roi_cy, SEARCH_R_LOCKED, W, H)
            search = ir_np[sy0:sy1, sx0:sx1]

            th, tw = template.shape
            if search.shape[0] >= th and search.shape[1] >= tw:
                search_blur = cv2.GaussianBlur(search, (3, 3), 0)
                templ_blur = cv2.GaussianBlur(template, (3, 3), 0)

                res = cv2.matchTemplate(search_blur, templ_blur, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                match_score = float(max_val)

                best_x = sx0 + max_loc[0] + tw // 2
                best_y = sy0 + max_loc[1] + th // 2

                cand_depth_med, _ = roi_depth_median(depth_raw, best_x, best_y, ROI_R, depth_scale)

                if expected_depth is None and cand_depth_med > 0:
                    expected_depth = cand_depth_med

                accept = False
                if match_score >= MATCH_THRESH and cand_depth_med > 0 and expected_depth is not None:
                    if abs(cand_depth_med - expected_depth) <= DEPTH_JUMP_MAX:
                        accept = True

                if accept:
                    candidate_cx, candidate_cy = best_x, best_y
                    expected_depth = (1 - DEPTH_ALPHA) * expected_depth + DEPTH_ALPHA * cand_depth_med
                # else reject drift

        roi_cx, roi_cy = candidate_cx, candidate_cy

        # ----------------------------
        # ROI STATS (current ROI)
        # ----------------------------
        depth_med, depth_valid_pct = roi_depth_median(depth_raw, roi_cx, roi_cy, ROI_R, depth_scale)
        ir_mean, ir_std, (x0, x1, y0, y1) = roi_stats(ir_np, roi_cx, roi_cy, ROI_R)
        conf_mean, _, _ = roi_stats(conf_np, roi_cx, roi_cy, ROI_R)

        if not locked:
            expected_depth = None

        # ----------------------------
        # TRIAL CAPTURE (press 'g')
        # ----------------------------
        if trial_active:
            ts = datetime.now().isoformat(timespec="milliseconds")
            note = ""
            if depth_valid_pct < MIN_VALID_PCT:
                note = "low_valid"

            # Per-frame row
            trial_frame_rows.append([
                run_id,
                trial_id,
                ts,
                condition,
                distance_label,
                f"{depth_med:.6f}",
                f"{depth_valid_pct:.2f}",
                f"{ir_mean:.6f}",
                f"{ir_std:.6f}",
                f"{conf_mean:.6f}",
                "" if match_score is None else f"{match_score:.6f}",
                "" if expected_depth is None else f"{expected_depth:.6f}",
                roi_cx,
                roi_cy,
                ROI_R,
                note
            ])

            # Accumulate stats for trial summary
            trial_stats["depth_med"].append(depth_med)
            trial_stats["depth_valid"].append(depth_valid_pct)
            trial_stats["ir_mean"].append(ir_mean)
            trial_stats["ir_std"].append(ir_std)
            trial_stats["conf_mean"].append(conf_mean)
            trial_stats["match"].append(-1.0 if match_score is None else match_score)

            trial_frames_left -= 1

            # If finished, write to disk
            if trial_frames_left <= 0:
                trial_active = False
                trial_end_ts = datetime.now().isoformat(timespec="seconds")

                # Write per-frame rows
                append_frames_rows(trial_frame_rows)

                # Compute summary stats
                def mean_std(arr):
                    arr = np.array(arr, dtype=np.float64)
                    return float(arr.mean()), float(arr.std(ddof=0))

                depth_med_mean, depth_med_std = mean_std(trial_stats["depth_med"])
                depth_valid_mean, _ = mean_std(trial_stats["depth_valid"])
                ir_mean_mean, ir_mean_std = mean_std(trial_stats["ir_mean"])
                ir_std_mean, _ = mean_std(trial_stats["ir_std"])
                conf_mean_mean, _ = mean_std(trial_stats["conf_mean"])
                match_mean, _ = mean_std(trial_stats["match"])

                notes = ""
                if condition == "unknown":
                    notes += "condition_unknown;"
                if not locked:
                    notes += "unlocked_during_trial;"
                if depth_valid_mean < MIN_VALID_PCT:
                    notes += "low_depth_valid_avg;"

                append_summary_row([
                    run_id,
                    trial_id,
                    trial_start_ts,
                    trial_end_ts,
                    condition,
                    distance_label,
                    TRIAL_FRAMES,
                    f"{depth_med_mean:.6f}",
                    f"{depth_med_std:.6f}",
                    f"{depth_valid_mean:.2f}",
                    f"{ir_mean_mean:.6f}",
                    f"{ir_mean_std:.6f}",
                    f"{ir_std_mean:.6f}",
                    f"{conf_mean_mean:.6f}",
                    f"{match_mean:.6f}",
                    notes
                ])

                print(f"\n✅ Trial {trial_id} saved. condition={condition}, distance={distance_label}m")
                print(f"   frames:  {frames_csv_path}")
                print(f"   summary: {summary_csv_path}\n")

                # Reset buffers
                trial_frame_rows = []
                trial_stats = {k: [] for k in trial_stats}

        # ----------------------------
        # DISPLAY (scaled canvas + banner)
        # ----------------------------
        disp_ir = cv2.equalizeHist(ir_np) if use_equalize else ir_np
        disp_ir_big = cv2.resize(disp_ir, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
        disp_bgr = cv2.cvtColor(disp_ir_big, cv2.COLOR_GRAY2BGR)

        status = "LOCKED" if locked else "UNLOCKED"
        box_color = (0, 255, 0) if locked else (0, 200, 255)

        sx0, sx1 = x0 * SCALE, x1 * SCALE
        sy0, sy1 = y0 * SCALE, y1 * SCALE
        cv2.rectangle(disp_bgr, (sx0, sy0), (sx1 - 1, sy1 - 1), box_color, 2)
        cv2.drawMarker(
            disp_bgr,
            (roi_cx * SCALE, roi_cy * SCALE),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2
        )

        banner = np.zeros((BANNER_H, W * SCALE, 3), dtype=np.uint8)

        line1 = f"{status}  (c=lock r=reset g=grab s=snap q=quit | 1/2/3 distance | w/d condition | e contrast)"
        line2 = f"condition={condition}   distance_label={distance_label}m   ROI_R={ROI_R}   scale={SCALE}x"
        if locked and expected_depth is not None:
            line3 = f"depth_med={depth_med:.3f}m  valid={depth_valid_pct:.1f}%  exp={expected_depth:.3f}m"
        else:
            line3 = f"depth_med={depth_med:.3f}m  valid={depth_valid_pct:.1f}%"
        line4 = f"IR_mean={ir_mean:.1f}  IR_std={ir_std:.1f}  conf_mean={conf_mean:.1f}"
        if locked:
            ms = -1.0 if match_score is None else match_score
            line5 = f"match={ms:.2f}  gate={DEPTH_JUMP_MAX:.2f}m  match_thresh={MATCH_THRESH:.2f}"
        else:
            line5 = "match=-- (unlock)  tracker inactive"

        if trial_active:
            line6 = f"TRIAL ACTIVE: id={trial_id}  frames_left={trial_frames_left}/{TRIAL_FRAMES}  -> saving to {os.path.basename(summary_csv_path)}"
        else:
            line6 = f"Ready. Press 'g' to record {TRIAL_FRAMES} frames into CSV (requires LOCKED for best results)."

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.75
        thk = 2
        y = 28
        for ln in [line1, line2, line3, line4, line5, line6]:
            cv2.putText(banner, ln, (10, y), font, fs, (255, 255, 255), thk, cv2.LINE_AA)
            y += 24

        vis = np.vstack([banner, disp_bgr])
        cv2.imshow(WINDOW_NAME, vis)

        # ----------------------------
        # KEY HANDLERS
        # ----------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('e'):
            use_equalize = not use_equalize

        if key == ord('w'):
            condition = "wet"

        if key == ord('d'):
            condition = "dry"

        if key in DISTANCE_KEYS:
            idx = DISTANCE_KEYS.index(key)
            distance_label = DISTANCE_LABELS[idx]

        if key == ord('r'):
            locked = False
            template = None
            expected_depth = None
            roi_cx, roi_cy = W // 2, H // 2
            print("\nReset/unlocked.")

        if key == ord('c'):
            # Lock template from current ROI location
            x0t, x1t, y0t, y1t = clamp_box(roi_cx, roi_cy, ROI_R, W, H)
            template = ir_np[y0t:y1t, x0t:x1t].copy()
            locked = True

            depth_med_now, _ = roi_depth_median(depth_raw, roi_cx, roi_cy, ROI_R, depth_scale)
            expected_depth = depth_med_now if depth_med_now > 0 else None

            print(f"\nLocked template at (x={roi_cx}, y={roi_cy}), template={template.shape}, expected_depth={expected_depth}")

        if key == ord('g'):
            # Start one trial capture
            if trial_active:
                print("\n⚠ Trial already active. Wait for it to finish.")
            else:
                trial_id += 1
                trial_active = True
                trial_frames_left = TRIAL_FRAMES
                trial_start_ts = datetime.now().isoformat(timespec="seconds")
                trial_frame_rows = []
                trial_stats = {k: [] for k in trial_stats}

                print(f"\n🎬 Starting trial {trial_id}: condition={condition}, distance_label={distance_label}m")
                if not locked:
                    print("   ⚠ You are UNLOCKED. Press 'c' to lock on rockwool for best results.")
                print(f"   Will record {TRIAL_FRAMES} frames then write summary row.\n")

        if key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            np.save(os.path.join(SNAP_DIR, f"{ts}_ir.npy"), ir_np)
            np.save(os.path.join(SNAP_DIR, f"{ts}_conf.npy"), conf_np)
            np.save(os.path.join(SNAP_DIR, f"{ts}_depth_raw.npy"), depth_raw)
            cv2.imwrite(os.path.join(SNAP_DIR, f"{ts}_preview.png"), vis)
            print(f"\nSaved snapshot: {ts} in {SNAP_DIR}/")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nStopped.")
    print(f"\nYour CSVs for this run:")
    print(f"  Frames:  {frames_csv_path}")
    print(f"  Summary: {summary_csv_path}")
