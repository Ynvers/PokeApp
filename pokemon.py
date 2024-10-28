import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image

with open("pikachu_or_rondoudou.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Welcome in PokeRecon")
st.write("""
            In this app, you can find which type of Pokémon you look like! \n
            Currently, only Pikachu and Jigglypuff (Rondoudou) are supported in this version.
         """)

st.header("Uplooad or Capture a Picture")
uploaded_file = st.file_uploader("Upload a picture")
captured_image = st.camera_input("Or take a picture")

if uploaded_file is not None or captured_image is not None:
    image = uploaded_file if uploaded_file is not None else captured_image
    img = Image.open(image)
    st.image(img, caption="Uploaded/Captured Image", use_column_width=True)
    
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    opencv_image_resized = np.expand_dims(cv2.resize(opencv_image,(200,200)), axis=0)
    
    res = model.predict(opencv_image_resized)
    predicted_class = np.argmax(res) 
    if predicted_class == 1:
        st.write("IT'S A PIKACHU !")
    elif predicted_class == 0 :
        st.write("IT'S A RONDOUDOU !")