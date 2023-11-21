import sys
import numpy as np
import pandas as pd
# from matplotlib.pyplot import imread
#from IPython.display import Image 

import tensorflow as tf
import tensorflow_hub as hub
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
print(tf.config.experimental.list_physical_devices('GPU'))
IMG_SIZE=224
INPUT_SHAPE = [None,IMG_SIZE,IMG_SIZE,3]

# set up data
data =pd.read_csv('labels.csv')
total_images  = len(data)
labels= np.array(data["kind"])
un_labels= np.unique(labels)
mapping = {i : un_labels[i] for i in range(0, len(un_labels))}
key_map= {un_labels[i] : i for i in range(0, len(un_labels))}
num_labels = np.vectorize(mapping.get)(labels)
OUTPUT_SHAPE = len(un_labels)

# load model
model = model = tf.keras.models.load_model('/home/haroon/Documents/Animal_detection/saved_models/model.h5',custom_objects={"KerasLayer":hub.KerasLayer})

def process_img(Path,size=[IMG_SIZE,IMG_SIZE]):
    """
    convert image at some path to tensor
    """
    
    img = tf.io.read_file(Path)
    img = tf.image.decode_image(img,channels=3,expand_animations = False)  
    img = tf.image.convert_image_dtype(img,tf.float32)  # scales values [0-1] if necessary
    img = tf.image.resize(img,size)
    return img

# img  = process_img('/home/haroon/Documents/Animal_detection/test_images/1.jpg')
# preds = model.predict(img, verbose =1)
# pred_label = np.argmax(preds)     

# create GUI
def app():
    
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200,200,500,500)  #xpos,ypos,width,height
    win.setWindowTitle("Animal Detection")
    win.show()
    sys.exit(app.exec_())
    
app()