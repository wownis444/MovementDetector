import d3dshot
import time
import viewer
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import *

beg = (100, 100)
width = 500//3
height = 280//3
padding = 10
i = 0
windows = []
app = QtWidgets.QApplication(sys.argv)
window = viewer.mymainwindow((beg[0] + (i * (width + padding)), beg[1]), (width, height))
window.show()
windows.append(window)
time.sleep(1.1)

th = d3dshot.d3dshot.Thread(capture_output="numpy", obszar=(width, height))
th.changePixmap.connect(window.setImage)
th.start()
app.exec_()

time.sleep(6000)  # Capture is non-blocking so we wait explicitely
