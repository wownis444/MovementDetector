import d3dshot
import time

th = d3dshot.d3dshot.Thread(capture_output="numpy")
th.start()

time.sleep(6000)  # Capture is non-blocking so we wait explicitely
