import streamlit as st
import numpy as np
import seaborn as sns
import tensorflow as tf
from config import find_project_root
from pathlib import Path

def load_model() -> tf.keras.Model:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    model = tf.keras.models.load_model(f"{PROJECT_ROOT}/src/image_recognition_network/models/image_classification_model.h5")
    return model

def load_image(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (32,32))
    img = tf.cast(img, tf.float32) / 255.0

    return img

def main():
    model = load_model()
    st.title("Image classification model")
    image = st.file_uploader("Upload your photo here.",accept_multiple_files=False)

    if image:
        img = load_image(image)
        prediction = model.predict(img)
        predictied_label = np.argmax(prediction)
        st.write(f"The uploaded image is that of a {predictied_label}, with a {np.max(prediction) * 100}% certainity.")
    else:
        st.write("There has been an issue with the uploaded file.")

if __name__ == "__main__":
    main()