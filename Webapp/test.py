import streamlit as st
from PIL import Image
import numpy as np
import cv2

import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model_path = "output_1/detector.h5"
binarizer_path = "output_1/lb.pickle"

@st.cache
def model_loader(model_path,binarizer_path):
    model = load_model(model_path)
    lb = pickle.loads(open(binarizer_path, "rb").read())
    return model, lb

def plot_bboxes2(image,pred_bbox,pred_label):
    x1,x2,y1,y2 = pred_bbox
    copy = image.copy()
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    R,G,B = 50,255,50
    color = (R/255,G/255,B/255)
    copy = cv2.putText(copy, pred_label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2) # label in black text
    copy = cv2.rectangle(copy,(x1,y1),(x2,y2),color,2) # predicted bbox in green
    return copy

### Excluding Imports ###
st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file!=None:
    # img = cv2.imdecode(np.fromstring(uploaded_file, np.uint8), 1)
    # img = cv2.imread(img)
    image = Image.open(uploaded_file)
    (w,h) = image.size # just the way PIL works - cv2 does h,w
    col1, col2, col3 = st.beta_columns(3)
    with col2:
        st.header("Uploaded Image.")
        st.image(image)

    image2 = image.resize((224,224))
    image2 = img_to_array(image2) / 255.0
    image2 = np.expand_dims(image2, axis=0)
    model, lb = model_loader(model_path,binarizer_path)
    (boxPreds, labelPreds) = model.predict(image2)
    (x1, y1, x2, y2) = boxPreds.flatten()
    #(h, w) = image2.shape[1:3] # (1,224,224,3)
    # re-scale the bounding box coordinates to fit the original image
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)
    index = np.argmax(labelPreds, axis=1)
    label = lb.classes_[index][0]
    image = img_to_array(image) / 255.0
    predict_image = plot_bboxes2(image,(x1,y1,x2,y2),label)
    with col2:
        st.header("Predicted Image.")
        st.image(predict_image)
