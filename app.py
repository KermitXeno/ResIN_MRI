# -*- coding: utf-8 -*-

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSplitter, QLabel, QVBoxLayout, QPushButton, QListWidget, QFileDialog, QListWidgetItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import tensorflow as tf
import numpy as np 
from PIL import Image
from PySide6.QtCore import Signal
from UtilsGNN.Model_Init import *
from UtilsGNN.Model_HF_Download import *
from Models.Model_Grad_Cam import GradCAM
import matplotlib.cm as cm

style = """

QWidget {
    background-color: #1d1d1d;
    color: #ededed;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 12px;
}

QLabel#header {
    font-size: 12px;
    font-weight: 600;
    color: #ffffff;
    padding: 4x;
    background-color: #1e1e1e;
}

QLabel {
    background: transparent;
}

QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    padding: 8px 14px;
    border-radius: 6px;
}

QPushButton:hover {
    background-color: #3a3a3a;
}

QPushButton:pressed {
    background-color: #454545;
}

QPushButton:checked {
    background-color: #1565C0;
    border: 1px solid #1e88e5;
}

QListWidget {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    padding: 4px;
}

QListWidget::item {
    padding: 6px;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #1565C0;
    color: white;
}

QListWidget::item:hover {
    background-color: #2a2a2a;
}
"""

class SectionFileSelect(QWidget):

    imageSelected = Signal(tf.Tensor, np.ndarray, str)

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, title: str):
        super().__init__()
        #DEFAULT MODEL
        self.mode = "resnet"

        #UI LAYOUT
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)

        self.openbtn = QPushButton("Select Folder")
        self.openbtn.clicked.connect(self.openfolder)

        self.listwidget = QListWidget()
        self.listwidget.itemSelectionChanged.connect(self.onselection)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.openbtn)
        layout.addWidget(self.listwidget)

        base = os.path.dirname(os.path.abspath(__file__))
        testdir = os.path.join(base, "UtilsGNN", "Test_Data")

        if os.path.exists(testdir):
            self.populate(testdir)
    #FOLDER SELECTION HANDLER
    def openfolder(self):
  
        base = os.path.dirname(os.path.abspath(__file__))
        defaultdir = os.path.join(base, "UtilsGNN", "Test_Data")


        if not os.path.exists(defaultdir):
            defaultdir = ""

        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", defaultdir, QFileDialog.ShowDirsOnly
        )

        if folder:
            self.populate(folder)

    def populate(self, folder: str):
        self.listwidget.clear()

        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            ext = os.path.splitext(name)[1].lower()

            if os.path.isfile(path) and ext in self.IMAGE_EXTS:
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, path)
                self.listwidget.addItem(item)
    
    #IMAGE SELECTION HANDLER
    def onselection(self):
        items = self.listwidget.selectedItems()
        if not items:
            return

        path = items[0].data(Qt.UserRole)

        tensor, display = self.preprocess(path)

        self.imageSelected.emit(tensor, display, path)

    def setmode(self, mode: str):
        self.mode = mode

    # IMAGE PREPROCESSING(SAME AS TRAINING) 
    def preprocess(self, path: str):
        img = Image.open(path).convert("RGB")
        img = img.resize((128, 128), Image.BILINEAR)

        displayimg = np.array(img)

        img = np.array(img, dtype = np.float32)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis = -1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis = -1)

        if self.mode == "resnet":
            img = tf.keras.applications.resnet_v2.preprocess_input(img)

        elif self.mode == "resin":
            pass

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        img = tf.expand_dims(img, axis = 0)

        return img, displayimg

class SectionModelSelect(QWidget):
    modelchanged = Signal(str)

    def __init__(self, title: str):
        super().__init__()

        #UI SETUP
        self.current_mode = "resnet"

        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedHeight(32) 

        self.btnresnet = QPushButton("ResNet50V2")
        self.btnresin = QPushButton("RELURes")
        
        self.btnresnet.setCheckable(True)
        self.btnresin.setCheckable(True)

        self.btnresnet.setChecked(True)

        #BUTTON ACTIONS
        self.btnresnet.clicked.connect(lambda: self.setmode("resnet"))
        self.btnresin.clicked.connect(lambda: self.setmode("resin"))

        #UI LAYOUT
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.btnresnet)
        layout.addWidget(self.btnresin)
    
    #SET MODE AND UPDATE BUTTON STATES
    def setmode(self, mode: str):
        self.current_mode = mode
        self.btnresnet.setChecked(mode == "resnet")
        self.btnresin.setChecked(mode == "resin")
        self.modelchanged.emit(mode)

class SectionOutput(QWidget):
    def __init__(self, title: str):
        super().__init__()

        #UI SETUP
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setObjectName("header")
        self.title.setFixedHeight(32)

        self.imagelabel = QLabel("GradCAM output")
        self.imagelabel.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        #BUTTON ACTIONS
        layout.addWidget(self.title)
        layout.addWidget(self.imagelabel)

    #CREATE HEATMAP OVERLAY AND DISPLAY
    def showcam(self, heatmap: np.ndarray, baseimg: np.ndarray):

        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap = tf.image.resize(heatmap[..., np.newaxis], (baseimg.shape[0], baseimg.shape[1])).numpy().squeeze()

        cmap = cm.get_cmap("jet")  
        colored = cmap(heatmap)  
        colored = (colored[..., :3] * 255).astype(np.uint8)

        overlay = (0.6 * baseimg + 0.4 * colored).astype(np.uint8)

        h, w, ch = overlay.shape
        bytesperline = ch * w

        qimg = QImage(overlay.data, w, h, bytesperline, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        self.imagelabel.setPixmap(pix.scaled(self.imagelabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #UI SETUP
        self.setWindowTitle("UI")
        self.resize(1200, 800)

        self.filesection = SectionFileSelect("Image Selection")
        self.modelsection = SectionModelSelect("Model Selection")
        self.rightsection = SectionOutput("Model Window")

        rsplitter = QSplitter(Qt.Horizontal)
        lsplitter = QSplitter(Qt.Vertical)

        lsplitter.addWidget(self.filesection)
        lsplitter.addWidget(self.modelsection)

        rsplitter.addWidget(lsplitter)
        rsplitter.addWidget(self.rightsection)

        rsplitter.setStretchFactor(0, 0)
        rsplitter.setStretchFactor(1, 1)  

        lsplitter.setStretchFactor(0, 5)  
        lsplitter.setStretchFactor(1, 0) 

        #MODEL LOGIC
        self.setCentralWidget(rsplitter)

        self.modelsection.modelchanged.connect(self.filesection.setmode)
        self.modelsection.modelchanged.connect(self.onmodelchanged)

        self.filesection.imageSelected.connect(self.rungradcam)

        self.resinrelu = None
        self.resinresnettrans = None
        self.gradcamresnet = None
        self.gradcamresin = None

        self.currentmode = "resnet"

    #EVENT TO CHANGE MODEL
    def onmodelchanged(self, mode: str):
        self.currentmode = mode
    
    #'0': Mild_Demented
    #'1': Moderate_Demented
    #'2': Non_Demented
    #'3': Very_Mild_Demented

    #RUN GRADCAM AND DISPLAY RESULTS(WITH SHOWCAM)
    def rungradcam(self, tensor, display, path):
        if self.currentmode == "resnet":
            cam = self.gradcamresnet
        else:
            cam = self.gradcamresin

        if cam is None:
            print("GradCAM not ready")
            return

        heatmap = cam(tensor)


        self.rightsection.showcam(heatmap, display)

if __name__ == "__main__":
    tpath, rpath = initialize()

    if not (os.path.exists(tpath) and os.path.exists(rpath)):
        download_keras_files(repoid = "KermitXeno/MRIBLandRESIN",localdir = MODELp)

    resinrelu = initRELURES(rpath)
    resinresnettrans = initRELUtrans(tpath)

    #comment out after selecting target layer
    resinrelu.summary()
    resinresnettrans.summary()

    resinrelu.trainable = False
    resinresnettrans.trainable = False

    app = QApplication(sys.argv)
    app.setStyleSheet(style)
    window = MainWindow()
    window.resinrelu = resinrelu
    window.resinresnettrans = resinresnettrans

    window.gradcamresnet = GradCAM(model = resinresnettrans, targetlayer = "conv3_block2_out")
    window.gradcamresin = GradCAM(model = resinrelu, targetlayer = "relu_inception_1" )

    window.show()
    sys.exit(app.exec())