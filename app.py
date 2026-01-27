# -*- coding: utf-8 -*-

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSplitter, QLabel, QVBoxLayout
from PySide6.QtCore import Qt


class Section(QWidget):

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

        left_top = Section("Image Selection")
        left_bottom = Section("Model Selection")

        left_splitter.addWidget(left_top)
        left_splitter.addWidget(left_bottom)
        left_splitter.setSizes([400, 400])

        right_section = Section("Model Window")

        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_section)
        main_splitter.setSizes([400, 800])

        self.setCentralWidget(main_splitter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())