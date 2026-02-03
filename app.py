# -*- coding: utf-8 -*-

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSplitter, QLabel, QVBoxLayout
from PySide6.QtCore import Qt
import UtilsGNN
from UtilsGNN.Model_Init import *
from UtilsGNN.Model_HF_Download import *

#todo add top bars to the windows, display titles of splitters on the bars and various tools

#todo get list of files from utilsgnn/test_data
class SectionFileSelect(QWidget):

    def __init__(self, title: str):
        super().__init__()

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
            }
            QLabel {
                color: #ddd;
                font-size: 16px;
            }
        """)

#todo add buttons for model options, check that the models exist, have button to load and unload them into memory.
class SectionModelSelect(QWidget):
    global resinrelu, resinresnettrans

    def __init__(self, title: str):
        super().__init__()

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
            }
            QLabel {
                color: #ddd;
                font-size: 16px;
            }
        """)

#todo give predictions from model, add overlay to show siggy of features in prediction over the image as a heat mapm, display the name of the model above the output. also displays the input image
class SectionOutput(QWidget):

    def __init__(self, title: str):
        super().__init__()

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(label)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
            }
            QLabel {
                color: #ddd;
                font-size: 16px;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MRI Analysis UI")
        self.resize(1200, 800)

        main_splitter = QSplitter(Qt.Horizontal)

        left_splitter = QSplitter(Qt.Vertical)

        left_top = SectionFileSelect("Image Selection")
        left_bottom = SectionModelSelect("Model Selection")

        left_splitter.addWidget(left_top)
        left_splitter.addWidget(left_bottom)
        left_splitter.setSizes([400, 400])

        right_section = SectionOutput("Model Window")

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_section)
        main_splitter.setSizes([400, 800])

        self.setCentralWidget(main_splitter)


if __name__ == "__main__":
    UtilsGNN.initialize

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())