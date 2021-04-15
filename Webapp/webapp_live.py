import streamlit as st
from PIL import Image
import os
import pandas as pd
from streamlit_cropper import st_cropper
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import scipy.spatial
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


upper_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_features.npy')
img_ids_upp = np.load('/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_feature_img_ids.npy')
lower_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Lower_InceptionResnet_features.npy')
img_ids_low = np.load('/home/ubuntu/capstone/final_inception_features/Lower_InceptionResnet_feature_img_ids.npy')
full_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Full_InceptionResnet_features.npy')
img_ids_full = np.load('/home/ubuntu/capstone/final_inception_features/Full_InceptionResnet_feature_img_ids.npy')
flipkart_commerce = pd.read_csv("/home/ubuntu/capstone/Data/flipkart_ecommerce.csv")

def display_image(results,img_id_cat):
    img_id = []
    scores = []
    uni_id = []
    unique_list = []
    for id, score in results[0:25]:
        img_id.append(img_id_cat[id])
        values = img_id_cat[id].split("/")[-1].split("_")[0].split(".")[0]
        uni_id.append("/".join(img_id_cat[id].split("/")[:-1]) + "/" + values + ".jpg")
        scores.append((1 - score) * 100)
    for li in uni_id:
        if li not in unique_list:
            unique_list.append(li)
    captions = []
    for i in unique_list:
        pid = i.split("/")[-1].split("_")[0].split(".")[0]
        captions.append(pid)
    st.markdown("#")
    st.header("Recommendations")
    st.markdown("#")
    st.image(unique_list[:6], width=250, caption=captions[:6])

st.title("Fashion Visual Search and Recommendation")
st.markdown("#")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file!=None:
    image = Image.open(uploaded_file)
    image = Image.open(uploaded_file)
    (w,h) = image.size
    st.header("Uploaded Image")
    cropped_img = st_cropper(image)
    st.header("Make use of the cropper - rectangular box to focus your area of interest")
    os.chdir("/home/ubuntu/capstone/darknet")
    cropped_img.save("/home/ubuntu/capstone/darknet/test_images.jpg")
    path = "/home/ubuntu/capstone/darknet/test_images.jpg"
    command_darknet = "./darknet detector test data/obj.data cfg/yolo-obj.cfg yolo-obj_final.weights -dont_show -thresh 0.3 -out result.json < test_sample.txt"
    os.chdir("/home/ubuntu/capstone/darknet")
    file_newpath = "echo " + path + " > test_sample.txt"
    os.system(file_newpath)
    os.system(command_darknet)
    results = pd.read_json("./result.json")
    for i in results['objects']:
        if len(i) > 1:
            st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")
        elif len(i) == 0:
            st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")
        elif len(i) == 1:
            for clas in i:
                if clas['confidence'] >= 0.70:
                    predict_image = Image.open(path)
                    center_x = clas['relative_coordinates']['center_x']
                    center_y = clas['relative_coordinates']['center_y']
                    wid = clas['relative_coordinates']['width']
                    heig = clas['relative_coordinates']['height']
                    width = wid / 2.0
                    height = heig / 2.0
                    img_wid = predict_image.size[0]
                    img_ht = predict_image.size[1]
                    left = (center_x - width) * img_wid
                    right = (center_x + width) * img_wid
                    bottom = (center_y + height) * img_ht
                    top = (center_y - height) * img_ht
                    predict_image = predict_image.crop((left, top, right, bottom))
                    predict_image.save("croped_image.jpg")
                    model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                    pooling='avg')
                    predict_image = Image.open('croped_image.jpg')
                    img = predict_image.resize((224, 224))
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    query_features = model.predict(x)
                    query_features = np.array(query_features).reshape((1, query_features.shape[1]))
                    if clas['name'] == 'Upper':
                        distances = scipy.spatial.distance.cdist(upper_ex_fea, query_features, "cosine")
                        results = zip(range(len(distances)), distances)
                        results = sorted(results, key=lambda x: x[1])
                        display_image(results,img_ids_upp)
                    elif clas['name'] == 'Lower':
                        distances = scipy.spatial.distance.cdist(lower_ex_fea, query_features, "cosine")
                        results = zip(range(len(distances)), distances)
                        results = sorted(results, key=lambda x: x[1])
                        display_image(results, img_ids_low)
                    elif clas['name'] == 'Full':
                        distances = scipy.spatial.distance.cdist(full_ex_fea, query_features, "cosine")
                        results = zip(range(len(distances)), distances)
                        results = sorted(results, key=lambda x: x[1])
                        display_image(results, img_ids_full)
                else:
                    st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")