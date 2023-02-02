import streamlit as st
import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import imutils


# load the image file with cv2 so it'll work with my fucntions:
def load_image(image_file):
    image_file.seek(0)
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1,224,224,3)
    return image

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    binary_model = keras.models.load_model('./binary_model.h5')
    multi_model = keras.models.load_model('./multi_model.h5')
    bi_prediction = binary_model.predict(image)
    mu_prediction = multi_model.predict(image)
    class_names={0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary tumor'}
    mu_class = np.argmax(mu_prediction)
    mu_class_cat = class_names[mu_class]
    
    st.image(Image.open(uploaded_file), caption = "Uploaded Image", use_column_width=False)
    if bi_prediction == 1 and mu_class_cat != 'no tumor':
        st.write(f"There may be cause for concern. This looks like {mu_class_cat}.")
        st.write(f"It looks like {mu_class_cat}.")
    elif bi_prediction == 0 and mu_class_cat != 'no tumor':
        st.write("Model is uncertain.")
        st.write(f"If there is something there, it looks like {mu_class_cat}")
    else:
        st.write("Everything looks fine.")
        
    bi_accuracy = binary_model.evaluate(image, bi_prediction, verbose=0)[1]
    mu_accuracy = multi_model.evaluate(image, mu_prediction, verbose=0)[1]
    st.write("{:.2f}% confidence.".format(bi_accuracy*100))
    st.write("Disclaimer: This is not a diagnostic tool. Please consult a medical professional for any health related inquiries.")