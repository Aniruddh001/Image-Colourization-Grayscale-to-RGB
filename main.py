from model import load,upload_predict
import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageOps

sns.set_theme(style="darkgrid")
sns.set()


st.title('Grayscale To Coloured Image')


# st.write('Upload your Grayscale Image U0001F447')
file = st.file_uploader('Upload your Grayscale Image ',type=["jpg","jpeg", "png"])

converter = load()

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    
    st.write("Grayscale Image")
    st.image(image)
    predictions = upload_predict(image, converter) 
    st.write("The Coloured Image is : ")
    st.image(Image.fromarray(np.uint8(predictions)).convert('RGB'))



