import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from config import find_project_root
from indices_map import idx_map

def load_model() -> tf.keras.Model:
    project_root = Path(__file__).resolve().parents[2]
    model_path = (
        project_root
        / "src"
        / "image_recognition_network"
        / "models"
        / "image_classification_model.h5"
    )
    return tf.keras.models.load_model(model_path)


def load_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()

    img = tf.io.decode_png(image_bytes, channels=3)
    img = tf.image.resize(img, (32, 32))
    img = tf.cast(img, tf.float32) / 255.0

    # Models expect batches: (batch, height, width, channels)
    return tf.expand_dims(img, axis=0)

def display_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

def main():
    model = load_model()

    st.title("Image Classification Model")
    uploaded_file = st.file_uploader(
        "Upload your photo here.",
        type=["png"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        display_image(uploaded_file)
        img = load_image(uploaded_file)
        prediction = model.predict(img, verbose=0)

        predicted_label = int(np.argmax(prediction[0]))
        for index, label in idx_map.items():
            if index == predicted_label:
                predicted_label = label
                break
        confidence = float(np.max(prediction[0]) * 100)

        st.write(
            f"Predicted class: {predicted_label}"
            f"(Confidence: {confidence:.2f}%)"
        )
    else:
        st.write("Upload an image to make a prediction.")

if __name__ == "__main__":
    main()