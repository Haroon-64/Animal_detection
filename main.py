import sys
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap 
from PyQt5.QtCore import Qt 
import tensorflow as tf
import tensorflow_hub as hub

IMG_SIZE=224
INPUT_SHAPE = [None,IMG_SIZE,IMG_SIZE,3]
path = ''
model_path = os.getcwd() + '/model.h5'
# set up data
data = pd.read_csv('labels.csv')
total_images = len(data)
labels = np.array(data["kind"])
un_labels = np.unique(labels)
mapping = {i : un_labels[i] for i in range(0, len(un_labels))}
key_map = {un_labels[i] : i for i in range(0, len(un_labels))}
num_labels = np.vectorize(mapping.get)(labels)
OUTPUT_SHAPE = len(un_labels)

# load model
model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})

def process_img(Path,size=[IMG_SIZE,IMG_SIZE]):
    """
    convert image at some path to tensor
    """
    img = tf.io.read_file(Path)
    img = tf.image.decode_image(img,channels=3,expand_animations = False)  
    img = tf.image.convert_image_dtype(img,tf.float32)  # scales values [0-1] if necessary
    img = tf.image.resize(img,size)
    return img


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Classifier App')
        self.setGeometry(100, 100, 800, 600)  # Set a fixed size for the window

        self.central_widget = QLabel(self)
        self.central_widget.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.central_widget)

        self.statusBar().showMessage('Ready')

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setGeometry(10, 10, 100, 30)

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")

        file_dialog.fileSelected.connect(self.process_loaded_image)
        file_dialog.exec_()

    def process_loaded_image(self, selected_file):
        if selected_file:
            self.image_path = selected_file
            img = process_img(self.image_path)
            preds = model.predict(np.expand_dims(img, axis=0), verbose=1)
            pred_label = np.argmax(preds)

            self.display_image(pred_label)

    def display_image(self, pred_label):
        img = QPixmap(self.image_path)

        # Resize the image to fit within the existing window size
        img = img.scaled(self.central_widget.size(), Qt.KeepAspectRatio)

        self.central_widget.setPixmap(img)

        label_text = f'Prediction: {un_labels[pred_label]}'
        self.statusBar().showMessage(label_text)

        # Enable the Load Image button for rechoosing another image
        self.load_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())