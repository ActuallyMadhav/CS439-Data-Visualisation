import sys
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QGridLayout, QComboBox, QSlider, QLabel

# Create some (random) datasets to select from
times = np.arange(365)
temps = 55 + 4.*np.random.standard_normal(365)
prices = 100 + 3.*np.random.standard_normal(365)
ratings = 2.5 + np.random.standard_normal(365)
ratings[ratings<0] = 0
ratings[ratings>5] = 5
prices[prices<25] = 25
prices[prices>200] = 200
data = { 'Temperatures': {'x': times, 'y': temps, 'xlabel': 'days', 'ylabel': 'Daily Max Temp (in F)'}, 
         'Prices': { 'x': times, 'y': prices, 'xlabel': 'days', 'ylabel': 'Price (in $)'},
         'Ratings': { 'x': times, 'y': ratings, 'xlabel': 'days', 'ylabel': 'Average Rating (out of 5)' } }

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.linewidth = 1

        self.mpl_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.mpl_canvas.setFixedWidth(600)
        self.mpl_canvas.setFixedHeight(600)

        self.dropdown = QComboBox(self.main_widget)
        for name in data.keys():
            self.dropdown.addItem(name)

        self.slider = QSlider(self.main_widget)
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(10)

        layout = QGridLayout(self.main_widget)
        layout.addWidget(NavigationToolbar(self.mpl_canvas, self), 0, 0, 1, 6)
        layout.addWidget(self.mpl_canvas,                          1, 0, 4, 4)
        layout.addWidget(QLabel('Data selection'),                 1, 4, 1, 1)
        layout.addWidget(self.dropdown,                            1, 5, 1, 1)
        layout.addWidget(QLabel('Line width'),                     2, 4, 1, 1)
        layout.addWidget(self.slider,                              2, 5, 1, 1)

        self.ax = self.mpl_canvas.figure.subplots()
        self.plots = self.create_plot(data['Temperatures'])

        self.dropdown.activated.connect(self.update_data)
        self.slider.valueChanged.connect(self.update_width)

    def create_plot(self, d):
        plots = self.ax.plot(d['x'], d['y'], linewidth=self.linewidth)
        self.ax.set_xlabel(d['xlabel'])
        self.ax.set_ylabel(d['ylabel'])
        return plots

    def update_data(self, index):
        name = self.dropdown.currentText()
        self.ax.clear()
        self.plots = self.create_plot(data[name])
        self.mpl_canvas.draw()

    def update_width(self, width):
        self.linewidth = width
        self.plots[0].set_linewidth(width)
        self.mpl_canvas.draw()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()