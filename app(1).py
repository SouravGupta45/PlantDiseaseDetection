import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import streamlit as st

default_image_size = tuple((128, 128))
model = tf.keras.models.load_model('CNN_model2.keras')

logo = Image.open('leaf.png')
st.image(logo,width=100,channels='BGR')

st.header('Plant Disease Detection App')

img_path = st.file_uploader('Upload Image:', type='jpg')
#image = Image.open(img_path)
#st.image(image,caption='Uploaded Image',width=300,channels='BGR')

#image = '/home/trillian/plantvillage/PlantVillage/Tomato_healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG'

def convert_image_to_array(image_dir):
    try:
        image = Image.open(image_dir)
        #st.image(image,channels='BGR')
        if image is not None :
            image = image.resize(default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict(img_loc):
    image = convert_image_to_array(img_loc)
    image_ = np.array([image], dtype=np.float16) / 225.0
    result = model.predict(image_)[0][0]
    if result>0.6:
        return 1
    else:
        return 0

if st.button('Submit'):
    if img_path:
        st.write('Processing image...')
        image = Image.open(img_path)
        st.image(image,caption='Uploaded Image',width=300,channels='BGR')

        result = predict(img_path)
        if(result==1):
            st.subheader('Your plant is detected to have a disease!\nUse adequate amount of pesticides or remove the plant to save other plants from the disease.\n\nStay Healthy! Stay Safe!')
        else:
            st.subheader('Congratulations! Your plant is healthy!\n We recommend you time to time checkup for timely decisons to keep your garden green and healthy\n\n Stay Healthy! Stay Safe!')
        #else:
            #st.write('Invalid image location. Not an image!')
    else:
        st.write('No path entered. Please try again!!')

