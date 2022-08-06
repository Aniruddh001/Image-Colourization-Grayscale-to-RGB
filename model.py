import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def load():
    model = load_model('Image_Colourization.h5')
    return model

def upload_predict(upload_image,model):
    size = (160,160)
    image = np.array(image)
    img_resize = cv2.resize(img, dsize=(160, 160),interpolation=cv2.INTER_CUBIC)
    img_reshape = img_resize.reshape(1,160,160,3)
    prediction = model.predict(image_reshape).reshape(160, 160,3)
    return prediction