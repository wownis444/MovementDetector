import json

import cv2

import base64
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import time
from PyQt5.QtCore import *


class mymainwindow(QtWidgets.QMainWindow):
    def __init__(self, position=(0, 0), size=(250, 140)):
        QtWidgets.QMainWindow.__init__(self)
        self.image = QLabel(self)
        self.user_name = QLabel(self)
        self.nick = ""
        self.size = size
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
            QtCore.QSize(self.size[0], self.size[1]),
            QtWidgets.qApp.desktop().availableGeometry()))

        self.move(position[0], position[1])

        self.image = QLabel(self)
        pixmap = QPixmap('image.jpeg')
        self.image.setPixmap(pixmap)
        self.image.resize(self.size[0], self.size[1])
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

# file = open("config.json")
# config = json.load(file)
# ips = config["ips"]
# port = config["port"]
#
# padding = 10
# beg = (100, 100)
#
# context = zmq.Context()
# app = QtWidgets.QApplication(sys.argv)
# threads = []
# sockets = []
# windows = []
# for i in range(len(ips)):
#     window = mymainwindow((beg[0] + (i * (width + padding)), beg[1]))
#     window.show()
#     windows.append(window)
#
#     time.sleep(0.1)
#
#     socket = context.socket(zmq.SUB)
#     socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
#     adress = 'tcp://' + ips[i] + ":" + str(port)
#     print(adress)
#     socket.connect(adress)
#     sockets.append(socket)
#
#     th = Thread(i)
#     th.changePixmap.connect(window.setImage)
#     th.changeUserName.connect(window.setUserName)
#     th.start()
#
# print("started")
# app.exec_()
