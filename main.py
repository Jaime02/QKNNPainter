import os

import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QPixmap, QPaintEvent, QMouseEvent, QColor, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy, \
    QLabel, QCheckBox
from matplotlib import pyplot as plt


class KNNWidget(QWidget):
    def __init__(self, main_window):
        QWidget.__init__(self)
        self.main_window = main_window

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

        # The widget should have the smallest possible size
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.label = QLabel("Dataset not loaded")
        self.main_layout.addWidget(self.label)

        self.dataset = np.ndarray(dtype=np.ndarray, shape=(10, 1000))

    @staticmethod
    def calculate_distance(pixmap: np.array, dataset_pixmap: np.array):
        # This function calculates the distance between the pixmap and the dataset
        # The distance is calculated by the manhattan distance using numpy
        return np.sum(np.abs(pixmap - dataset_pixmap))

    def perform_knn(self):
        if not self.main_window.dataset_loaded:
            return

        pixmap = self.main_window.pixel_painting_widget.pixmap.toImage()
        pixmap = np.array([b for b in pixmap.convertToFormat(QImage.Format_Grayscale8).bits().tobytes()])

        # Calculate the distance between the pixmap and the dataset
        distances = np.zeros(10000)
        for i, row in enumerate(self.dataset):
            for j, array in enumerate(row):
                distance = self.calculate_distance(pixmap, array)
                distances[i*1000+j] = distance

        # Get the index of the smallest distance
        plt.figure()
        plt.plot(distances)
        plt.show()


class PixelPaintingWidget(QWidget):
    def __init__(self, main_window):
        QWidget.__init__(self)
        self.main_window = main_window
        self.pixmap = QPixmap(28, 28)
        self.pixmap.fill(Qt.white)

        self.painter = QPainter(self.pixmap)
        self.pen = QPen(QColor(0, 0, 0))
        self.pen.setWidth(3)
        self.painter.setPen(self.pen)

    def save_png(self):
        self.pixmap.save("test.png")

    def clear_pixmap(self):
        self.pixmap.fill(Qt.white)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def resizeEvent(self, e):
        QWidget.resizeEvent(self, e)
        a = min(self.width(), self.height())
        self.resize(a, a)

    def mousePressEvent(self, event: QMouseEvent):
        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        QWidget.mouseMoveEvent(self, event)
        current_pos = event.position().toPoint()
        current_pixel = QPoint(current_pos.x() / self.width() * 28, current_pos.y() / self.height() * 28)
        self.painter.setRenderHints(QPainter.Antialiasing, True)
        self.painter.drawPoint(current_pixel)
        self.update()

        if self.main_window.perform_knn.isChecked() and self.main_window.dataset_loaded:
            self.main_window.knn_widget.perform_knn()


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle("QKnnPainter")

        self.dataset_loaded = False

        cw = QWidget()
        self.main_layout = QVBoxLayout()
        cw.setLayout(self.main_layout)
        self.setCentralWidget(cw)

        self.buttons_layout = QHBoxLayout()
        self.main_layout.addLayout(self.buttons_layout)

        self.second_row_layout = QHBoxLayout()
        self.main_layout.addLayout(self.second_row_layout)

        self.load_dataset_button = QPushButton("Load dataset")
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.buttons_layout.addWidget(self.load_dataset_button)

        self.clear_button = QPushButton("Clear")
        self.buttons_layout.addWidget(self.clear_button)

        self.save_png_button = QPushButton("Save as PNG")
        self.buttons_layout.addWidget(self.save_png_button)

        self.perform_knn_button = QPushButton("Perform KNN")
        self.perform_knn_button.setDisabled(True)
        self.buttons_layout.addWidget(self.perform_knn_button)

        self.knn_widget = KNNWidget(self)
        self.second_row_layout.addWidget(self.knn_widget)

        self.perform_knn = QCheckBox("Perform KNN automatically")
        # self.perform_knn.setChecked(True)
        self.second_row_layout.addWidget(self.perform_knn, 0, Qt.AlignRight)
        self.perform_knn_button.clicked.connect(self.knn_widget.perform_knn)

        self.pixel_painting_widget = PixelPaintingWidget(self)
        self.pixel_painting_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.clear_button.clicked.connect(self.pixel_painting_widget.clear_pixmap)
        self.save_png_button.clicked.connect(self.pixel_painting_widget.save_png)

        self.main_layout.addWidget(self.pixel_painting_widget)

    def load_dataset(self):
        self.dataset_loaded = True
        self.perform_knn_button.setEnabled(True)
        if os.path.exists("dataset.npy"):
            self.knn_widget.dataset = np.load("dataset.npy", allow_pickle=True)
            self.knn_widget.label.setText("Dataset loaded")
            return

        for number in range(10):
            with open(f"binary_data/data{number}.bin", "rb") as f:
                for img in range(1000):
                    b = f.read(28*28)
                    self.knn_widget.dataset[number][img] = np.array([i for i in b])
                    self.knn_widget.dataset[number][img].reshape(28, 28)
        np.save("dataset.npy", self.knn_widget.dataset)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    window.resize(800, 600)
    app.exec()
