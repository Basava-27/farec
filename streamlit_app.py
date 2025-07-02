import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
picture_size = 48

st.title("Facial Expression Recognition (FER)")
st.write("Upload a face image and the model will predict the expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.convert("L")
    image = image.resize((picture_size, picture_size))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    prediction = model.predict(image_array)
    predicted_label = expression_labels[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"**{predicted_label}**")

    st.subheader("Confidence:")
    confidence = {expression_labels[i]: float(prediction[0][i]) for i in range(len(expression_labels))}
    st.bar_chart(confidence)
