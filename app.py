##################################################################################################
#                                  IMPORTING NECESSARY LIBRARIES                                 #
##################################################################################################

import tensorflow as tf
import streamlit as st
import numpy as np
import os
import gdown

import base64

from io import BytesIO

from PIL import Image

from tensorflow.keras.utils import load_img, img_to_array

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnetb0_preprocess

##################################################################################################
#                                     LOADING BEST MODELS                                        #
##################################################################################################

# Creating folder to store Models
MODEL_DIR = "Final Trained models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Mapping Best Models to Google Drive File ID's
MODEL_FILE_ID = {"CNN" : "1W-F3GIcbCYwuYmmvUr5YgFyM4GDLDHY4",
                 "VGG16" : "1Vm5Bg4MA8PEWl6nwf827icvRJnLYjriz",
                 "ResNet50" : "1czkcuvAa9l-lCcA799FBokZRqly_vu1t",
                 "MobileNet" : "1FoRPuA6depPgzhPJDFRzSt7kA7fnKJk6",
                 "InceptionV3" : "1yy-8AgwFoZSX4E6DL12o3jJxXRCsD_ab",
                 "EfficientNetB0" : "1xPE83GmUocxiBCf4Bq6W_imwC9MyXXXp"}

MODEL_PATHS = {}

for name, file_id in MODEL_FILE_ID.items():
    model_path = os.path.join(MODEL_DIR, F"best_{name}_model.keras")
    download_url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {name} model..."):
            gdown.download(download_url, model_path, quiet=False)
            st.success(f"{name} model downloaded!")
    
    MODEL_PATHS[name] = model_path

MODEL_CFGS = {"CNN" : {"size" : (224, 224), "preprocess" : Rescaling(1./255)},
              "VGG16" : {"size" : (224, 224), "preprocess" : vgg16_preprocess},
              "ResNet50" : {"size" : (224, 224), "preprocess" : resnet50_preprocess},
              "MobileNet" : {"size" : (224, 224), "preprocess" : mobilenet_preprocess},
              "InceptionV3" : {"size" : (224, 224), "preprocess" : inceptionv3_preprocess},
              "EfficientNetB0" : {"size" : (224, 224), "preprocess" : efficientnetb0_preprocess}}

CLASS_NAMES = ["animal fish", "animal fish bass", "fish sea_food black_sea_sprat", "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel", "fish sea_food red_mullet", "fish sea_food red_sea_bream", "fish sea_food sea_bass", "fish sea_food shrimp", "fish sea_food striped_red_mullet", "fish sea_food trout"]

# Cache saves the loaded model and saves time while predicting different images
@st.cache_resource
def load_models(path):
     return load_model(path)

# Loading Models into Dictionary
models = {name : load_models(path) for name, path in MODEL_PATHS.items()}

# Preprocessing Method
def preprocess_image(image, target_size, preprocess_func):
    # Load and Resize and accepts RGB images
    img = load_img(image, target_size=target_size)

    # Convert to array
    img_array = img_to_array(img)

    # Add batch dimensions 
    img_array = tf.expand_dims(img_array, axis=0)

    # Model-specific preprocess
    img_array = preprocess_func(img_array)

    return img_array
        
##################################################################################################
#                                     STREAMLIT PAGE CONFIGS                                     #
##################################################################################################

st.set_page_config(layout="wide")

st.markdown("<h2 style='color:cyan; text-align:center;'>🐟 Multi Class Fish Image Classification 🌊</h2>", 
            unsafe_allow_html=True)
st.markdown("")

st.sidebar.title("Fish Classifier")
page = st.sidebar.radio("", ["🐡 Home", "🔮 Predict Fish Class"])

if page == "🐡 Home":
    logo_path = "trail_background.jpg"
    image = Image.open(logo_path)
    st.image(image, use_container_width=True)
    st.markdown("")
    st.markdown(
        """<p style='text-align: justify;'>
            Fish image classification streamlit application aims at classifying user-uploaded images into different categories using 
            <a href="https://www.geeksforgeeks.org/deep-learning/introduction-deep-learning/" target="_blank" style="color:blue;">
            Deep Learning
            </a>
             - 
            <a href="https://www.geeksforgeeks.org/computer-vision/computer-vision/" target="_blank" style="color:blue;">
            Computer Vision
            </a>
             models include
            <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network" target="_blank" style="color:blue;">
            CNN
            </a>
             built from scratch and pre-trained 
            <a href="https://en.wikipedia.org/wiki/Transfer_learning" target="_blank" style="color:blue;">
            Transfer learning 
            </a>
             models like 
            <a href="https://keras.io/api/applications/vgg/" target="_blank" style="color:blue;">
            VGG16
            </a>
            , 
            <a href="https://keras.io/api/applications/resnet/" target="_blank" style="color:blue;">
                ResNet50
            </a>
            , 
            <a href="https://keras.io/api/applications/mobilenet/" target="_blank" style="color:blue;">
                MobileNet
            </a>
            , 
            <a href="https://keras.io/api/applications/inceptionv3/" target="_blank" style="color:blue;">
                InceptionV3
            </a>
             and 
            <a href="https://keras.io/api/applications/efficientnet/" target="_blank" style="color:blue;">
                EfficientNetB0
            </a>
            to improve accuracy of the model. The trained models are saved for reuse and this pplication lets you upload and predict fish images in real-time.
        </p>""",unsafe_allow_html=True)
    
    st.markdown("")

    st.subheader("🐠 Fish Categories")
    with st.container(border=True):
        for i in range(0,len(CLASS_NAMES)):
            st.markdown(f"{i}. {CLASS_NAMES[i]}")


elif page == "🔮 Predict Fish Class":
    upload = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if upload:
        st.success("🔗 Image Upload Successful.")
        st.subheader("✅ Uploaded Image")
        with st.container():
            # Open uploaded image
            uploaded_im = Image.open(upload)

            # Convert Image to Base64
            buffered = BytesIO()
            uploaded_im1 = uploaded_im.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Image View via HTML
            st.markdown(f"""
                <div style="width:400px; height:200px; border:1px solid gray; padding:5px;">
                    <img src="data:image/jpeg;base64,{img_str}"  
                    style="width:100%; height:100%; object-fit: contain;">
                </div>
                """, unsafe_allow_html=True)

        selected_model = st.sidebar.radio("🤖 Models:", ["All", "CNN", "VGG16", "ResNet50", "MobileNet", "InceptionV3", "EfficientNetB0"])

        if selected_model == "All":
            st.subheader("🔍 Predictions per Model")

            all_predictions = []

            for name, model in models.items():

                # Loading config and preprocessing for each models
                cfg = MODEL_CFGS[name]
                preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

                # Predict 
                preds = model.predict(preprocessed_img, verbose=0)
                all_predictions.append(preds)

                pred_class = np.argmax(preds, axis=1)[0]
                st.markdown(f"**{name}** Prediction : 👉 {CLASS_NAMES[pred_class]}  \n📈 Confidence Score :  ({preds[0][pred_class]:.4f})")

            # All Model Average Prediction
            combined_pred = np.mean(all_predictions, axis=0)
            combined_class = np.argmax(combined_pred, axis=1)[0]

            st.subheader("🎯 Combined Prediction")
            st.success(f"Final prediction : **{CLASS_NAMES[combined_class]}**")
            st.write(f"🌡️ Overall Confidence Score : {combined_pred[0][combined_class]*100:.2f} %")

        elif selected_model == "CNN":
            st.subheader(f"🔍 {selected_model} Model Prediction")
            
            cfg = MODEL_CFGS["CNN"]
            preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

            # Predict 
            pred = models["CNN"].predict(preprocessed_img, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
            st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

        elif selected_model == "VGG16":
            st.subheader(f"🔍 {selected_model} Model Prediction")
            
            cfg = MODEL_CFGS["VGG16"]
            preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

            # Predict 
            pred = models["VGG16"].predict(preprocessed_img, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
            st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

        elif selected_model == "ResNet50":
            st.subheader(f"🔍 {selected_model} Model Prediction")
            
            cfg = MODEL_CFGS["ResNet50"]
            preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

            # Predict 
            pred = models["ResNet50"].predict(preprocessed_img, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
            st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

        elif selected_model == "MobileNet":
            st.subheader(f"🔍 {selected_model} Model Prediction")
            
            cfg = MODEL_CFGS["MobileNet"]
            preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

            # Predict 
            pred = models["MobileNet"].predict(preprocessed_img, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
            st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

        # elif selected_model == "InceptionV3":
        #     st.subheader(f"🔍 {selected_model} Model Prediction")
            
        #     cfg = MODEL_CFGS["InceptionV3"]
        #     preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

        #     # Predict 
        #     pred = models["InceptionV3"].predict(preprocessed_img, verbose=0)
        #     pred_class = np.argmax(pred, axis=1)[0]
        #     st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
        #     st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

        # elif selected_model == "EfficientNetB0":
        #     st.subheader(f"🔍 {selected_model} Model Prediction")
            
        #     cfg = MODEL_CFGS["EfficientNetB0"]
        #     preprocessed_img = preprocess_image(upload, cfg["size"], cfg["preprocess"])

        #     # Predict 
        #     pred = models["EfficientNetB0"].predict(preprocessed_img, verbose=0)
        #     pred_class = np.argmax(pred, axis=1)[0]
        #     st.success(f"Predicted Class : 👉 **{CLASS_NAMES[pred_class]}**")
        #     st.write(f"🌡️ Confidence Score : {pred[0][pred_class]*100:.2f} %")

