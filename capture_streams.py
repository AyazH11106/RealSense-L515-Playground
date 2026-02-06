"""
L515 IR + Confidence + Depth
- Depth-gated template tracking (press 'c' to lock ROI on rockwool)
- Trial logging for “wet vs dry separability up to X cm”
- AUTO ROI size adjustment using depth_valid% (only when LOCKED)

KEYS
  c : lock (capture template) / start tracking
  r : reset / unlock
  1/2/3 : set distance label (default: 0.50 / 0.60 / 0.70)
  w/d : set condition wet/dry
  g : grab a TRIAL (record N frames -> save ONE row to SUMMARY CSV)
  f : toggle saving per-frame CSV (default OFF)
  s : snapshot (saves arrays + preview image)
  e : toggle contrast (visual only)
  a : toggle AUTO ROI on/off
  [ / ] : manual ROI shrink/grow
  q : quit

Outputs
  - summary CSV: ONE row per trial (what you want for quick plots)
  - frames CSV: optional, OFF by default (turn on with 'f')
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

ROI_R = 6
SEARCH_R_LOCKED = 50
MATCH_THRESH = 0.45

DEPTH_JUMP_MAX = 0.20
DEPTH_ALPHA = 0.15

DISTANCE_LABELS = ["0.50", "0.60", "0.70"]
DISTANCE_KEYS = [ord('1'), ord('2'), ord('3')]

TRIAL_FRAMES = 120
MIN_VALID_PCT = 60.0

WINDOW_NAME = "L515 IR (tracking + trials)"
SCALE = 3
BANNER_H = 165

OUT_DIR = "runs"
SNAP_DIR = os.path.join(OUT_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
frames_csv_path  = os.path.join(OUT_DIR, f"{run_id}_frames.csv")
summary_csv_path = os.path.join(OUT_DIR, f"{run_id}_summary.csv")

# ----------------------------
# AUTO ROI SIZE
# ----------------------------
AUTO_ROI_DEFAULT = True

ROI_R_MIN = 3
ROI_R_MAX = 14

ROI_VALID_TARGET = 88.0          # slightly lower target reduces “grow then shrink” loops
ROI_SHRINK_THRESH = 82.0         # shrink if smoothed valid% goes below this
ROI_GROW_THRESH   = 94.0         # grow if smoothed valid% goes above this

ROI_VALID_SMOOTH_ALPHA = 0.12    # slower smoothing = less twitchy

ROI_R_STEP = 1
ROI_R_CHECK_EVERY = 6            # evaluate every N frames
ROI_R_COOLDOWN = 30              # after resize, wait N frames before resizing again
ROI_R_PERSIST_N = 3              # require condition to hold for N checks

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
    x0, x1, y0, y1 = clamp_box(cx, cy, r, img_u8.shape[1], img_u8.shape[0])
    roi = img_u8[y0:y1, x0:x1].astype(np.float32)
    return float(roi.mean()), float(roi.std()), (x0, x1, y0, y1)

def ensure_csv_headers(save_frames: bool):
    os.makedirs(OUT_DIR, exist_ok=True)

    if save_frames and (not os.path.exists(frames_csv_path)):
        with open(frames_csv_path, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow([
                "run_id","trial_id","timestamp","condition","distance_label_m",
                "depth_med_m","depth_valid_pct","ir_mean","ir_std","conf_mean",
                "match_score","expected_depth_m","roi_cx","roi_cy","roi_r","auto_roi","note"
            ])

    if not os.path.exists(summary_csv_path):
        with open(summary_csv_path, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow([
                "run_id","trial_id","timestamp_start","timestamp_end",
                "condition","distance_label_m","n_frames",
                "depth_med_mean","depth_med_std",
                "depth_valid_mean","depth_valid_std",
                "ir_mean_mean","ir_mean_std",
                "ir_std_mean","ir_std_std",
                "conf_mean_mean","conf_mean_std",
                "match_mean","match_std",
                "roi_r_mean","roi_r_std",
                "auto_roi","notes"
            ])

def append_frames_rows(rows):
    with open(frames_csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)

def append_summary_row(row):
    with open(summary_csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def mean_std(arr):
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))

# ----------------------------
# MAIN
# ----------------------------
save_frames_csv = False  # ✅ default OFF (so you don't get tons of rows)
ensure_csv_headers(save_frames_csv)

try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W * SCALE + 40, H * SCALE + BANNER_H + 80)

    for _ in range(30):
        pipeline.wait_for_frames()

    locked = False
    roi_cx, roi_cy = W // 2, H // 2
    template = None
    expected_depth = None
    use_equalize = False

    roi_r = ROI_R
    auto_roi = AUTO_ROI_DEFAULT
    valid_smooth = None
    frame_counter = 0

    roi_cooldown = 0 #anti jitter stuff
    grow_hits = 0
    shrink_hits = 0

    condition = "unknown"
    distance_label = DISTANCE_LABELS[0]

    trial_active = False
    trial_id = 0
    trial_frames_left = 0
    trial_start_ts = None

    # Trial buffers
    trial_frame_rows = []
    trial_buf = {
        "depth_med": [],
        "depth_valid": [],
        "ir_mean": [],
        "ir_std": [],
        "conf_mean": [],
        "match": [],
        "roi_r": []
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

        # -------- tracking ----------
        match_score = None
        candidate_cx, candidate_cy = roi_cx, roi_cy

        if locked and template is not None:
            sx0, sx1, sy0, sy1 = clamp_box(roi_cx, roi_cy, SEARCH_R_LOCKED, W, H)
            search = ir_np[sy0:sy1, sx0:sx1]

            th, tw = template.shape
            if search.shape[0] >= th and search.shape[1] >= tw:
                res = cv2.matchTemplate(
                    cv2.GaussianBlur(search, (3, 3), 0),
                    cv2.GaussianBlur(template, (3, 3), 0),
                    cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                match_score = float(max_val)

                best_x = sx0 + max_loc[0] + tw // 2
                best_y = sy0 + max_loc[1] + th // 2

                cand_depth_med, _ = roi_depth_median(depth_raw, best_x, best_y, roi_r, depth_scale)
                if expected_depth is None and cand_depth_med > 0:
                    expected_depth = cand_depth_med

                accept = (
                    match_score >= MATCH_THRESH and
                    cand_depth_med > 0 and
                    expected_depth is not None and
                    abs(cand_depth_med - expected_depth) <= DEPTH_JUMP_MAX
                )

                if accept:
                    candidate_cx, candidate_cy = best_x, best_y
                    expected_depth = (1 - DEPTH_ALPHA) * expected_depth + DEPTH_ALPHA * cand_depth_med

        roi_cx, roi_cy = candidate_cx, candidate_cy

        # -------- ROI stats ----------
        depth_med, depth_valid_pct = roi_depth_median(depth_raw, roi_cx, roi_cy, roi_r, depth_scale)
        ir_mean, ir_std, (x0, x1, y0, y1) = roi_stats(ir_np, roi_cx, roi_cy, roi_r)
        conf_mean, _, _ = roi_stats(conf_np, roi_cx, roi_cy, roi_r)

        # -------- auto ROI ----------
        frame_counter += 1

        # smooth valid %
        if valid_smooth is None:
            valid_smooth = depth_valid_pct
        else:
            valid_smooth = (1 - ROI_VALID_SMOOTH_ALPHA) * valid_smooth + ROI_VALID_SMOOTH_ALPHA * depth_valid_pct

        # cooldown timer
        if roi_cooldown > 0:
            roi_cooldown -= 1

        # Only auto-adjust when LOCKED
        if locked and auto_roi and (frame_counter % ROI_R_CHECK_EVERY == 0) and (roi_cooldown == 0):
            # count “persistence” hits
            if valid_smooth < ROI_SHRINK_THRESH:
                shrink_hits += 1
                grow_hits = 0
            elif valid_smooth > ROI_GROW_THRESH:
                grow_hits += 1
                shrink_hits = 0
            else:
                # inside deadband -> reset both
                grow_hits = 0
                shrink_hits = 0

            # apply resize only if persistent for N checks
            if shrink_hits >= ROI_R_PERSIST_N and roi_r > ROI_R_MIN:
                roi_r -= ROI_R_STEP
                shrink_hits = 0
                roi_cooldown = ROI_R_COOLDOWN

            elif grow_hits >= ROI_R_PERSIST_N and roi_r < ROI_R_MAX:
                roi_r += ROI_R_STEP
                grow_hits = 0
                roi_cooldown = ROI_R_COOLDOWN

        # if unlocked, reset
        if not locked:
            valid_smooth = None
            expected_depth = None
            roi_cooldown = 0
            grow_hits = 0
            shrink_hits = 0

        # -------- TRIAL capture: store in buffers; SAVE ONE ROW at end ----------
        if trial_active:
            note = "low_valid" if depth_valid_pct < MIN_VALID_PCT else ""

            # Optional per-frame CSV rows (OFF by default)
            if save_frames_csv:
                ts = datetime.now().isoformat(timespec="milliseconds")
                trial_frame_rows.append([
                    run_id, trial_id, ts, condition, distance_label,
                    f"{depth_med:.6f}", f"{depth_valid_pct:.2f}",
                    f"{ir_mean:.6f}", f"{ir_std:.6f}", f"{conf_mean:.6f}",
                    "" if match_score is None else f"{match_score:.6f}",
                    "" if expected_depth is None else f"{expected_depth:.6f}",
                    roi_cx, roi_cy, roi_r, 1 if auto_roi else 0, note
                ])

            # Always accumulate stats for summary row
            trial_buf["depth_med"].append(depth_med)
            trial_buf["depth_valid"].append(depth_valid_pct)
            trial_buf["ir_mean"].append(ir_mean)
            trial_buf["ir_std"].append(ir_std)
            trial_buf["conf_mean"].append(conf_mean)
            trial_buf["match"].append(-1.0 if match_score is None else match_score)
            trial_buf["roi_r"].append(roi_r)

            trial_frames_left -= 1

            if trial_frames_left <= 0:
                trial_active = False
                trial_end_ts = datetime.now().isoformat(timespec="seconds")

                # Write frames only if enabled
                if save_frames_csv:
                    ensure_csv_headers(save_frames_csv)
                    append_frames_rows(trial_frame_rows)

                # Compute summary stats
                d_mean, d_std = mean_std(trial_buf["depth_med"])
                v_mean, v_std = mean_std(trial_buf["depth_valid"])
                irm_mean, irm_std = mean_std(trial_buf["ir_mean"])
                irstd_mean, irstd_std = mean_std(trial_buf["ir_std"])
                c_mean, c_std = mean_std(trial_buf["conf_mean"])
                m_mean, m_std = mean_std(trial_buf["match"])
                rr_mean, rr_std = mean_std(trial_buf["roi_r"])

                notes = ""
                if condition == "unknown":
                    notes += "condition_unknown;"
                if not locked:
                    notes += "unlocked_during_trial;"
                if v_mean < MIN_VALID_PCT:
                    notes += "low_depth_valid_avg;"
                if auto_roi:
                    notes += "auto_roi_on;"

                append_summary_row([
                    run_id, trial_id, trial_start_ts, trial_end_ts,
                    condition, distance_label, TRIAL_FRAMES,
                    f"{d_mean:.6f}", f"{d_std:.6f}",
                    f"{v_mean:.2f}", f"{v_std:.2f}",
                    f"{irm_mean:.6f}", f"{irm_std:.6f}",
                    f"{irstd_mean:.6f}", f"{irstd_std:.6f}",
                    f"{c_mean:.6f}", f"{c_std:.6f}",
                    f"{m_mean:.6f}", f"{m_std:.6f}",
                    f"{rr_mean:.3f}", f"{rr_std:.3f}",
                    1 if auto_roi else 0,
                    notes
                ])

                print(f"\n✅ Trial {trial_id} saved -> SUMMARY ONLY (frames_csv={'ON' if save_frames_csv else 'OFF'})")
                print(f"   summary: {summary_csv_path}")
                if save_frames_csv:
                    print(f"   frames : {frames_csv_path}")
                print("")

                # reset trial buffers
                trial_frame_rows = []
                trial_buf = {k: [] for k in trial_buf}

        # -------- DISPLAY ----------
        disp_ir = cv2.equalizeHist(ir_np) if use_equalize else ir_np
        disp_ir_big = cv2.resize(disp_ir, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
        disp_bgr = cv2.cvtColor(disp_ir_big, cv2.COLOR_GRAY2BGR)

        status = "LOCKED" if locked else "UNLOCKED"
        box_color = (0, 255, 0) if locked else (0, 200, 255)

        sx0, sx1 = x0 * SCALE, x1 * SCALE
        sy0, sy1 = y0 * SCALE, y1 * SCALE
        cv2.rectangle(disp_bgr, (sx0, sy0), (sx1 - 1, sy1 - 1), box_color, 2)
        cv2.drawMarker(disp_bgr, (roi_cx * SCALE, roi_cy * SCALE),
                       (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        
        smth_txt = "--" if valid_smooth is None else f"{valid_smooth:.1f}"
        banner = np.zeros((BANNER_H, W * SCALE, 3), dtype=np.uint8)
        line1 = f"{status} (c lock | r reset | g grab | f frames_csv | s snap | q quit | 1/2/3 dist | w/d cond | a autoROI | [ ] ROI | e contrast)"
        line2 = f"cond={condition}  dist={distance_label}m  ROI_R={roi_r}  autoROI={'ON' if auto_roi else 'OFF'}  frames_csv={'ON' if save_frames_csv else 'OFF'}"
        line3 = f"depth_med={depth_med:.3f}m  valid={depth_valid_pct:.1f}% (smth={smth_txt}% )"
        line4 = f"IR_mean={ir_mean:.1f}  IR_std={ir_std:.1f}  conf_mean={conf_mean:.1f}"
        if locked:
            ms = -1.0 if match_score is None else match_score
            line5 = (
                f"match={ms:.2f}  gate={DEPTH_JUMP_MAX:.2f}m  "
                f"autoROI: shrink<{ROI_SHRINK_THRESH:.0f}% grow>{ROI_GROW_THRESH:.0f}%  "
                f"cool={roi_cooldown}"
            )
        else:
            line5 = "match=-- tracker inactive"



        if trial_active:
            line6 = f"TRIAL ACTIVE: id={trial_id}  frames_left={trial_frames_left}/{TRIAL_FRAMES}"
        else:
            line6 = f"Ready: press 'g' to save ONE summary row (set w/d + 1/2/3 + lock 'c')"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thk, y = 0.70, 2, 28
        for ln in [line1, line2, line3, line4, line5, line6]:
            cv2.putText(banner, ln, (10, y), font, fs, (255, 255, 255), thk, cv2.LINE_AA)
            y += 24

        vis = np.vstack([banner, disp_bgr])
        cv2.imshow(WINDOW_NAME, vis)

        # -------- KEYS ----------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('e'):
            use_equalize = not use_equalize
        if key == ord('a'):
            auto_roi = not auto_roi
            print(f"\nAuto ROI is now {'ON' if auto_roi else 'OFF'}.")
        if key == ord('f'):
            save_frames_csv = not save_frames_csv
            ensure_csv_headers(save_frames_csv)
            print(f"\nframes CSV saving is now {'ON' if save_frames_csv else 'OFF'}.")

        if key == ord('['):
            roi_r = max(ROI_R_MIN, roi_r - 1)
        if key == ord(']'):
            roi_r = min(ROI_R_MAX, roi_r + 1)

        if key == ord('w'):
            condition = "wet"
        if key == ord('d'):
            condition = "dry"
        if key in DISTANCE_KEYS:
            distance_label = DISTANCE_LABELS[DISTANCE_KEYS.index(key)]

        if key == ord('r'):
            locked = False
            template = None
            expected_depth = None
            roi_cx, roi_cy = W // 2, H // 2
            roi_r = ROI_R
            print("\nReset/unlocked.")

        if key == ord('c'):
            x0t, x1t, y0t, y1t = clamp_box(roi_cx, roi_cy, roi_r, W, H)
            template = ir_np[y0t:y1t, x0t:x1t].copy()
            locked = True
            depth_now, _ = roi_depth_median(depth_raw, roi_cx, roi_cy, roi_r, depth_scale)
            expected_depth = depth_now if depth_now > 0 else None
            valid_smooth = depth_valid_pct
            print(f"\nLocked template at (x={roi_cx}, y={roi_cy}), template={template.shape}, expected_depth={expected_depth}")

        if key == ord('g'):
            if trial_active:
                print("\n⚠ Trial already active.")
            else:
                trial_id += 1
                trial_active = True
                trial_frames_left = TRIAL_FRAMES
                trial_start_ts = datetime.now().isoformat(timespec="seconds")
                trial_frame_rows = []
                trial_buf = {k: [] for k in trial_buf}
                print(f"\n🎬 Trial {trial_id} started: cond={condition}, dist={distance_label}m  (frames_csv={'ON' if save_frames_csv else 'OFF'})")

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
    print(f"  Summary: {summary_csv_path}")
    if save_frames_csv:
        print(f"  Frames:  {frames_csv_path}")
