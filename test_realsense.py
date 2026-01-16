import pyrealsense2 as rs
import time
# run this everytime .\.venv\Scripts\activate
# then run python list_devices.py

pipeline = rs.pipeline()
config = rs.config()

# L515 depth stream
config.enable_stream(rs.stream.depth)

print("Starting camera...")
profile = pipeline.start(config)

try:
    for i in range(60):  # ~2 seconds worth at 30fps
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue

        # read depth at center pixel
        w = depth.get_width()
        h = depth.get_height()
        d = depth.get_distance(w // 2, h // 2)

        print(f"Frame {i:02d} | center depth = {d:.3f} m")
        time.sleep(0.03)

finally:
    pipeline.stop()
    print("Stopped.")
