import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print("Number of devices found:", len(devices))
for dev in devices:
    print("Name:", dev.get_info(rs.camera_info.name))
    print("Serial:", dev.get_info(rs.camera_info.serial_number))
    print("FW:", dev.get_info(rs.camera_info.firmware_version))
    print("-" * 30)