import threading
import collections

import gc
import os
import time

from d3dshot.display import Display
from d3dshot.capture_output import CaptureOutput, CaptureOutputs

import base64
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import time
from PyQt5.QtCore import *


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self,
                 capture_output=CaptureOutputs.PIL,
                 frame_buffer_size=60,
                 pil_is_available=True,
                 numpy_is_available=False,
                 pytorch_is_available=False,
                 pytorch_gpu_is_available=False,
                 obszar=(200, 100)
                 ):
        super().__init__()
        self.displays = None
        self.detect_displays()

        self.display = None

        for display in self.displays:
            if display.is_primary:
                self.display = display
                break
        capture_output = CaptureOutputs.NUMPY
        self.capture_output = CaptureOutput(backend=capture_output)

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = collections.deque(list(), self.frame_buffer_size)

        self.previous_screenshot = None

        self.obszar = obszar
        self.region = None

        self._pil_is_available = pil_is_available
        self._numpy_is_available = numpy_is_available
        self._pytorch_is_available = pytorch_is_available
        self._pytorch_gpu_is_available = pytorch_gpu_is_available

        self._capture_thread = None
        self._is_capturing = False

    def _reset_displays(self):
        self.displays = list()

    def detect_displays(self):
        self._reset_displays()
        self.displays = Display.discover_displays()

    def _validate_region(self, region):
        region = region or self.region or None

        if region is None:
            return None

        if isinstance(region, list):
            region = tuple(region)

        if not isinstance(region, tuple) or len(region) != 4:
            raise AttributeError("'region' is expected to be a 4-length tuple")

        valid = True

        for i, value in enumerate(region):
            if not isinstance(value, int):
                valid = False
                break

            if i == 2:
                if value <= region[0]:
                    valid = False
                    break
            elif i == 3:
                if value <= region[1]:
                    valid = False
                    break

        if not valid:
            raise AttributeError(
                """Invalid 'region' tuple. Make sure all values are ints and that 'right' and 
                'bottom' values are greater than their 'left' and 'top' counterparts"""
            )

        return region

    def wyrownaj_obraz(self, image):
        shape = image.shape
        new_image = np.zeros_like(image)
        for i in range(314, 314 + 140):
            new_image[i, :] = np.roll(image[i, :], (-i * 6))

        return new_image

    def cut_region(self, image, region):
        shp = image.shape
        beg_y = (image.shape[0] // 2) - (region[0] // 2)
        width = (beg_y, beg_y + region[0])
        beg_x = (image.shape[1] // 2) - (region[1] // 2)
        height = (beg_x, beg_x + region[1])

        return image[width[0]:width[1], height[0]:height[1]]

    def process_image(self, image):
        # diff = cv2.absdiff(first_frame, frame)
        # diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        # _, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        # first_frame = frame
        # print("ASDASD")
        ret = cv2.resize(image, (self.obszar[1] * 2, self.obszar[0] * 2))
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # ret = cv2.filter2D(ret, -1, kernel)
        return ret

    def run(self):
        target_fps = 35
        frame_time = 1 / target_fps
        print("aaaaa")
        try:
            i = 0
            while True:
                cycle_start = time.time()
                region = None
                frame = self.display.capture(
                    self.capture_output.process,
                    region=self._validate_region(region)
                )
                if frame is not None:
                    # obszar = (140//2,250//2)
                    # frame = self.wyrownaj_obraz(frame)
                    frame = self.cut_region(frame, self.obszar)
                    print(frame.shape)
                    # frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
                    if i == 0:
                        first_frame = frame
                    else:
                        img = self.process_image(frame)

                        # cv2.imshow("adad", img)

                        # img = base64.b64decode(frame)
                        # npimg = np.fromstring(img, dtype=np.uint8)
                        source = img
                        h, w, ch = source.shape
                        bytesPerLine = 3 * w
                        cv2.cvtColor(source, cv2.COLOR_BGR2RGB, source)
                        print(source.shape)
                        convertToQtFormat = QtGui.QImage(source.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                        self.changePixmap.emit(convertToQtFormat)

                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                gc.collect()

                cycle_end = time.time()
                frame_time_left = frame_time - (cycle_end - cycle_start)

                if frame_time_left > 0:
                    time.sleep(frame_time_left)

                i += 1
        except Exception as e:
            print(e)

        cv2.destroyAllWindows()
        self._is_capturing = False


class D3DShot:

    def __init__(
            self,
            capture_output=CaptureOutputs.PIL,
            frame_buffer_size=60,
            pil_is_available=True,
            numpy_is_available=False,
            pytorch_is_available=False,
            pytorch_gpu_is_available=False,
    ):
        self.displays = None
        self.detect_displays()

        self.display = None

        for display in self.displays:
            if display.is_primary:
                self.display = display
                break

        self.capture_output = CaptureOutput(backend=capture_output)

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = collections.deque(list(), self.frame_buffer_size)

        self.previous_screenshot = None

        self.region = None

        self._pil_is_available = pil_is_available
        self._numpy_is_available = numpy_is_available
        self._pytorch_is_available = pytorch_is_available
        self._pytorch_gpu_is_available = pytorch_gpu_is_available

        self._capture_thread = None
        self._is_capturing = False

    @property
    def is_capturing(self):
        return self._is_capturing

    def get_latest_frame(self):
        return self.get_frame(0)

    def get_frame(self, frame_index):
        if frame_index < 0 or (frame_index + 1) > len(self.frame_buffer):
            return None

        return self.frame_buffer[frame_index]

    def get_frames(self, frame_indices):
        frames = list()

        for frame_index in frame_indices:
            frame = self.get_frame(frame_index)

            if frame is not None:
                frames.append(frame)

        return frames

    def get_frame_stack(self, frame_indices, stack_dimension=None):
        if stack_dimension not in ["first", "last"]:
            stack_dimension = "first"

        frames = self.get_frames(frame_indices)

        return self.capture_output.stack(frames, stack_dimension)

    def screenshot(self, region=None):
        region = self._validate_region(region)

        if self.previous_screenshot is None:
            frame = None

            while frame is None:
                frame = self.display.capture(self.capture_output.process, region=region)

            self.previous_screenshot = frame
            return frame
        else:
            for _ in range(300):
                frame = self.display.capture(self.capture_output.process, region=region)

                if frame is not None:
                    self.previous_screenshot = frame
                    return frame

            return self.previous_screenshot

    def screenshot_to_disk(self, directory=None, file_name=None, region=None):
        directory = self._validate_directory(directory)
        file_name = self._validate_file_name(file_name)

        file_path = f"{directory}/{file_name}"

        frame = self.screenshot(region=region)

        frame_pil = self.capture_output.to_pil(frame)
        frame_pil.save(file_path)

        return file_path

    def frame_buffer_to_disk(self, directory=None):
        directory = self._validate_directory(directory)

        for i, frame in enumerate(self.frame_buffer):
            frame_pil = self.capture_output.to_pil(frame)
            frame_pil.save(f"{directory}/{i + 1}.png")

    def capture(self, target_fps=60, region=None):
        target_fps = self._validate_target_fps(target_fps)

        if self.is_capturing:
            return False

        self._is_capturing = True

        app = QtWidgets.QApplication(sys.argv)
        th = Thread(1)
        th.start()
        print("poszlo")

        # self._capture_thread = threading.Thread(target=self._capture, args=(target_fps, region))
        # self._capture_thread.start()

        return True

    def screenshot_every(self, interval, region=None):
        if self.is_capturing:
            return False

        interval = self._validate_interval(interval)

        self._is_capturing = True

        self._capture_thread = threading.Thread(target=self._screenshot_every, args=(interval, region))
        self._capture_thread.start()

        return True

    def screenshot_to_disk_every(self, interval, directory=None, region=None):
        if self.is_capturing:
            return False

        interval = self._validate_interval(interval)
        directory = self._validate_directory(directory)

        self._is_capturing = True

        self._capture_thread = threading.Thread(target=self._screenshot_to_disk_every,
                                                args=(interval, directory, region))
        self._capture_thread.start()

        return True

    def stop(self):
        if not self.is_capturing:
            return False

        self._is_capturing = False
        self._capture_thread = None

        return True

    def benchmark(self):
        print(f"Preparing Benchmark...")
        print("")
        print(f"Capture Output: {self.capture_output.backend.__class__.__name__}")
        print(f"Display: {self.display}")
        print("")

        frame_count = 0

        start_time = time.time()
        end_time = start_time + 60

        print("Capturing as many frames as possible in the next 60 seconds... Go!")

        while time.time() <= end_time:
            self.screenshot()
            frame_count += 1

        print(f"Done! Results: {round(frame_count / 60, 3)} FPS")

    def detect_displays(self):
        self._reset_displays()
        self.displays = Display.discover_displays()

    def _reset_displays(self):
        self.displays = list()

    def _reset_frame_buffer(self):
        self.frame_buffer = collections.deque(list(), self.frame_buffer_size)

    def _validate_region(self, region):
        region = region or self.region or None

        if region is None:
            return None

        if isinstance(region, list):
            region = tuple(region)

        if not isinstance(region, tuple) or len(region) != 4:
            raise AttributeError("'region' is expected to be a 4-length tuple")

        valid = True

        for i, value in enumerate(region):
            if not isinstance(value, int):
                valid = False
                break

            if i == 2:
                if value <= region[0]:
                    valid = False
                    break
            elif i == 3:
                if value <= region[1]:
                    valid = False
                    break

        if not valid:
            raise AttributeError(
                """Invalid 'region' tuple. Make sure all values are ints and that 'right' and 
                'bottom' values are greater than their 'left' and 'top' counterparts"""
            )

        return region

    def _validate_target_fps(self, target_fps):
        if not isinstance(target_fps, int) or target_fps < 1:
            raise AttributeError(f"'target_fps' should be an int greater than 0")

        return target_fps

    def _validate_directory(self, directory):
        if directory is None or not isinstance(directory, str):
            directory = "."

        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

        return directory

    def _validate_file_name(self, file_name):
        if file_name is None or not isinstance(file_name, str):
            file_name = f"{time.time()}.png"

        file_extension = file_name.split(".")[-1]

        if file_extension not in ["png", "jpg", "jpeg"]:
            raise AttributeError("'file_name' needs to end in .png, .jpg or .jpeg")

        return file_name

    def _validate_interval(self, interval):
        if isinstance(interval, int):
            interval = float(interval)

        if not isinstance(interval, float) or interval < 1.0:
            raise AttributeError("'interval' should be one of (int, float) and be >= 1.0")

        return interval

    def _capture(self, target_fps, region):
        self._reset_frame_buffer()

        frame_time = 1 / target_fps

        while self.is_capturing:
            cycle_start = time.time()

            frame = self.display.capture(
                self.capture_output.process,
                region=self._validate_region(region)
            )

            if frame is not None:
                self.frame_buffer.appendleft(frame)


            else:
                if len(self.frame_buffer):
                    self.frame_buffer.appendleft(self.frame_buffer[0])

            gc.collect()

            cycle_end = time.time()

            frame_time_left = frame_time - (cycle_end - cycle_start)

            if frame_time_left > 0:
                time.sleep(frame_time_left)

        self._is_capturing = False

    def _screenshot_every(self, interval, region):
        self._reset_frame_buffer()

        while self.is_capturing:
            cycle_start = time.time()

            frame = self.screenshot(region=self._validate_region(region))
            self.frame_buffer.appendleft(frame)

            cycle_end = time.time()

            time_left = interval - (cycle_end - cycle_start)

            if time_left > 0:
                time.sleep(time_left)

        self._is_capturing = False

    def _screenshot_to_disk_every(self, interval, directory, region):
        while self.is_capturing:
            cycle_start = time.time()

            self.screenshot_to_disk(directory=directory, region=self._validate_region(region))

            cycle_end = time.time()

            time_left = interval - (cycle_end - cycle_start)

            if time_left > 0:
                time.sleep(time_left)

        self._is_capturing = False
