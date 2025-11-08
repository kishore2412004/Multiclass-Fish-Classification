##################################################################################################
#                                  IMPORTING NECESSARY LIBRARIES                                 #
##################################################################################################
import tensorflow as tf
import streamlit as st
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_pre
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_pre

##################################################################################################
#                                     MODEL CONFIGURATIONS                                       #
##################################################################################################
MODEL_PATHS = {
    "CNN": "/home/kishore/Multiclass-image-classification/Trained Models/best_model_CNN.keras",
    "VGG16": "/home/kishore/Multiclass-image-classification/Trained Models/best_vgg16_model_finetuned.keras",
    "ResNet50": "/home/kishore/Multiclass-image-classification/Trained Models/best_resnet50_model_finetuned.keras",
    "MobileNet": "/home/kishore/Multiclass-image-classification/Trained Models/best_resnet50_model_finetuned.keras",
    "InceptionV3": "/home/kishore/Multiclass-image-classification/Trained Models/best_inceptionv3_model_finetuned.keras"
}

MODEL_CFGS = {
    "CNN": {"size": (224, 224), "preprocess": Rescaling(1. / 255)},
    "VGG16": {"size": (224, 224), "preprocess": vgg16_pre},
    "ResNet50": {"size": (224, 224), "preprocess": resnet50_pre},
    "MobileNet": {"size": (224, 224), "preprocess": mobilenet_pre},
    "InceptionV3": {"size": (224, 224), "preprocess": inceptionv3_pre},
}

CLASS_NAMES = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

##################################################################################################
#                                     HELPER FUNCTIONS                                           #
##################################################################################################
@st.cache_resource
def load_cached_model(path):
    return load_model(path)

models = {name: load_cached_model(path) for name, path in MODEL_PATHS.items()}

def auto_enhance_image(image: Image.Image):
    """Auto-enhance brightness and contrast for unseen/Google images"""
    image = ImageOps.exif_transpose(image)  # Fix orientation
    image = image.convert("RGB")
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(1.2)  # Slight brightness
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(1.2)
    return image

def preprocess_image(image, target_size, preprocess_func):
    """Prepares uploaded image for model prediction"""
    image = auto_enhance_image(image)
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    return img_array

##################################################################################################
#                                     STREAMLIT CONFIG                                            #
##################################################################################################
st.set_page_config(layout="wide", page_title="🐟 Fish Classifier")

st.markdown("<h2 style='color:cyan; text-align:center;'>🐟 Multi-Class Fish Image Classification 🌊</h2>", unsafe_allow_html=True)
st.sidebar.title("Fish Classifier")

page = st.sidebar.radio("", ["🐡 Home", "🔮 Predict Fish Class"])

##################################################################################################
#                                         HOME PAGE                                              #
##################################################################################################
if page == "🐡 Home":
    st.write("""
    ### Welcome 🎣
    This app classifies uploaded fish images into multiple species using pre-trained deep learning models
    (CNN, VGG16, ResNet50, MobileNet, InceptionV3).
    
    The app automatically adjusts brightness, contrast, and orientation for custom or Google images 
    to ensure accurate predictions — no fine-tuning required.
    """)

    st.subheader("🐠 Fish Categories")
    for i, cls in enumerate(CLASS_NAMES, start=1):
        st.markdown(f"**{i}. {cls}**")

##################################################################################################
#                                      PREDICTION PAGE                                           #
##################################################################################################
elif page == "🔮 Predict Fish Class":
    upload = st.sidebar.file_uploader("📤 Upload Fish Image", type=["jpg", "jpeg", "png"])

    if upload:
        st.success("✅ Image uploaded successfully!")

        uploaded_image = Image.open(upload).convert("RGB")
        buffered = BytesIO()
        uploaded_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(f"""
        <div style="width:400px; height:250px; border:1px solid gray; margin:auto; border-radius:8px;">
            <img src="data:image/jpeg;base64,{img_str}" style="width:100%; height:100%; object-fit:contain;">
        </div>
        """, unsafe_allow_html=True)

        selected_model = st.sidebar.radio("🤖 Select Model:", ["All", *MODEL_PATHS.keys()])

        if selected_model == "All":
            all_predictions = []
            st.subheader("📊 Model-wise Predictions")

            for name, model in models.items():
                cfg = MODEL_CFGS[name]
                preprocessed_img = preprocess_image(uploaded_image, cfg["size"], cfg["preprocess"])
                preds = model.predict(preprocessed_img, verbose=0)
                all_predictions.append(preds)

                pred_class = np.argmax(preds, axis=1)[0]
                confidence = preds[0][pred_class] * 100
                st.markdown(f"**{name} →** 🐟 `{CLASS_NAMES[pred_class]}` | 🌡️ *{confidence:.2f}%*")

            combined_pred = np.mean(all_predictions, axis=0)
            combined_class = np.argmax(combined_pred, axis=1)[0]
            st.success(f"🏆 Final Combined Prediction → **{CLASS_NAMES[combined_class]}** ({combined_pred[0][combined_class]*100:.2f}%)")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(CLASS_NAMES, combined_pred[0])
            ax.set_title("Combined Confidence Across Classes")
            st.pyplot(fig)

        else:
            st.subheader(f"🔍 {selected_model} Model Prediction")
            cfg = MODEL_CFGS[selected_model]
            preprocessed_img = preprocess_image(uploaded_image, cfg["size"], cfg["preprocess"])
            preds = models[selected_model].predict(preprocessed_img, verbose=0)

            pred_class = np.argmax(preds, axis=1)[0]
            confidence = preds[0][pred_class] * 100

            st.success(f"🎯 Predicted Class: **{CLASS_NAMES[pred_class]}**")
            st.write(f"🌡️ Confidence: **{confidence:.2f}%**")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(CLASS_NAMES, preds[0])
            ax.set_title(f"{selected_model} - Confidence Distribution")
            st.pyplot(fig)
    else:
        st.info("📸 Please upload a fish image to begin prediction.")

##################################################################################################
#                                       END OF SCRIPT                                            #
##################################################################################################
