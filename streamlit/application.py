import streamlit as st
import keras
import os
from PIL import Image
import numpy as np
import cv2
import imutils
import shutil
import tempfile
from image_processing import crop_images 

user_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

#create a temporary direcory to store the st file for transformations
# file path
temp_dir = os.path.join(os.getcwd(), "temp")
# if the file path doesn't exist, create it
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)
    
    # only runs if there's an image file 
if user_image is not None:

    file_path = os.path.join(temp_dir, 'user_image.jpg')
    with open(file_path, 'wb') as f:
        f.write(user_image.getvalue())
        
    # cv2 read in the image for the transformation fn
    image1 = cv2.imread(os.path.join(temp_dir, 'user_image.jpg'))
    image = cv2.resize(image1, dsize = (224, 224), interpolation=cv2.INTER_CUBIC)
    cropped_image = crop_images([image])[0]
    # export the cropped image to temp folder
    cv2.imwrite(os.path.join(temp_dir, 'cropped_image.jpg'), cropped_image)
    # in order for the fn to work, I need it as an np array
    # initialize array
    np_image = []
    # read in the image file with cv2
    image = cv2.imread(os.path.join(temp_dir, 'cropped_image.jpg'))
    # resize it to what VGG16 is expecting
    image = cv2.resize(image,(224, 224))
    # append to array
    np_image.append(image)
    # redeclare the array as variable to use in model
    image = np.array(np_image)
    
    binary_model = keras.models.load_model('./models/binary_model.h5')
    multi_model = keras.models.load_model('./models/multi_model.h5')
    bi_prediction = binary_model.predict(image)
    mu_prediction = multi_model.predict(image)
    class_names={0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary tumor'}
    mu_class = np.argmax(mu_prediction)
    mu_class_cat = class_names[mu_class]
    st.image(Image.open(user_image), caption = "Uploaded Image", use_column_width=False)
    
    if bi_prediction < 0.5:
        st.write(f"The image provided appears to have no indication of a tumor.")
    else:
        # st.write(f"The image provided appears to have cause for concern.")
        if mu_class_cat != 'no tumor':
            st.write(f"The image is similar to those with {mu_class_cat}")
        else:
            st.write(f"There's a tumor, but I can't identify it at this time.")

    st.write("Disclaimer: This is not a diagnostic tool. Please consult a medical professional for any health related inquiries.")
    
shutil.rmtree(temp_dir)