import json

import cv2
import zmq
import base64
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import time
from PyQt5.QtCore import *

width = 250
height = 140
status_icon = "●"
statuses = {"connected": "'green'",
            "outofwindow": "'yellow'",
            "disconnected": "'red'"
            }


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeUserName = pyqtSignal(str)

    def __init__(self, i):
        super().__init__()
        self.id = i

    def run(self):
        frame = sockets[self.id].setsockopt(zmq.RCVTIMEO, 2000)
        while True:
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)

            h, w, ch = source.shape
            bytesPerLine = ch * w
            convertToQtFormat = QtGui.QImage(source.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)


class mymainwindow(QtWidgets.QMainWindow):
    def __init__(self, position=(0, 0)):
        QtWidgets.QMainWindow.__init__(self)
        self.image = QLabel(self)
        self.user_name = QLabel(self)
        self.nick = ""

        self.shown = True
        self.moving = False
        self.clickXOffset = 0
        self.clickYOffset = 0
        self.currentXPosOffset = 0
        self.currentYPosOffset = 0
        self.posX = 0
        self.posY = 0

        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(QtWidgets.QStyle.alignedRect(
            QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
            QtCore.QSize(width + 50, height + 50),
            QtWidgets.qApp.desktop().availableGeometry()))

        self.move(position[0], position[1])

        self.image = QLabel(self)
        pixmap = QPixmap('image.jpeg')
        self.image.setPixmap(pixmap)
        self.image.resize(width, height)
        self.image.move(0, 22)

        self.user_name.setText("<font color='red'>● Wownis </font>")

    def mousePressEvent(self, event):
        self.moving = True
        self.clickXOffset = event.x()
        self.clickYOffset = event.y()
        pass
        # QtWidgets.qApp.qui t

    def mouseReleaseEvent(self, event):
        self.moving = False

    def mouseMoveEvent(self, event):
        if self.moving:
            self.currentXPosOffset = event.globalX() - self.clickXOffset
            self.currentYPosOffset = event.globalY() - self.clickYOffset
            self.move(self.currentXPosOffset, self.currentYPosOffset)
        pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Backspace:
            if self.shown:
                self.move(self.currentXPosOffset, self.currentYPosOffset)
                self.shown = False
            else:
                self.move(self.currentXPosOffset - 1000, self.currentYPosOffset)
                self.shown = True
        event.accept()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.image.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def setUserName(self, text):

        splited = text.split(" ", 1)
        print(splited)
        status = splited[0]
        if status != "disconnected":
            nick = splited[1]
            self.nick = status_icon + " " + nick

        text = "<font color=" + statuses[status] + ">" + self.nick + "</font>"
        print(text)
        self.user_name.setText(text)



file = open("config.json")
config = json.load(file)
ips = config["ips"]
port = config["port"]

padding = 10
beg = (100, 100)

context = zmq.Context()
app = QtWidgets.QApplication(sys.argv)
threads = []
sockets = []
windows = []
for i in range(len(ips)):
    window = mymainwindow((beg[0] + (i * (width + padding)), beg[1]))
    window.show()
    windows.append(window)

    time.sleep(0.1)

    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    adress = 'tcp://' + ips[i] + ":" + str(port)
    print(adress)
    socket.connect(adress)
    sockets.append(socket)

    th = Thread(i)
    th.changePixmap.connect(window.setImage)
    th.changeUserName.connect(window.setUserName)
    th.start()

print("started")
app.exec_()
