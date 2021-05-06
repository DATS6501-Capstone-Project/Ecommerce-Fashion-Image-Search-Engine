# Load the necessary files
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

# Function to load the images to the webapp
def load_img_pattern(unique_list,sco,img_id,uni_id,flipkart_commerce,myntra_commerce):
    cols = st.beta_columns(3)
    for ind,i in enumerate(unique_list):
        pid = i.split("/")[-1].split("_")[0].split(".")[0]
        flip = flipkart_commerce[flipkart_commerce['pid'] == pid]['product_url']
        myn = myntra_commerce[myntra_commerce['uniq_id']==pid]['link']
        img = Image.open(img_id[uni_id.index(i)])
        img = img.resize((200, 300), Image.ANTIALIAS)
        img.save('resized_image.jpg')
        if not flip.empty:
            flip = flip.iloc[0]
            cols[ind].image("resized_image.jpg", use_column_width=True)
            cols[ind].markdown(f'''
                <a  href="{flip}"><p class = 'image_ref' >Flipkart</p></a>''',unsafe_allow_html=True)
            cols[ind].markdown(f'<p class="image_ref">Score: {sco[ind]}</p>', unsafe_allow_html=True)
        else:
            myn = myn.iloc[0]
            cols[ind].image("resized_image.jpg", use_column_width=True)
            cols[ind].markdown(f'''
                <a  href="{myn}"><p class = 'image_ref' >Myntra</p></a>''',unsafe_allow_html=True)
            cols[ind].markdown(f'<p class="image_ref">Score: {sco[ind]}</p>', unsafe_allow_html=True)

# Function to load the top scored similar images based on threshold set using similarity threshold filter
def display_image(results, img_id_cat,similarity_score,flipkart_commerce, myntra_commerce,met_eval):
    img_id = []
    scores = []
    uni_id = []
    unique_list = []
    sco = []
    for id, score in results:
        img_id.append(img_id_cat[id])
        values = img_id_cat[id].split("/")[-1].split("_")[0].split(".")[0]
        uni_id.append("/".join(img_id_cat[id].split("/")[:-1]) + "/" + values + ".jpg")
        if met_eval == 'cosine':
            scores.append((1 - score) * 100)
        elif met_eval == 'euclidean':
            scores.append((100 - score))
    sim_score = sum(map(lambda x: x >= similarity_score, scores))
    for li, scor in zip(uni_id, scores):
        if li not in unique_list:
            unique_list.append(li)
            sco.append(scor)
    sim_score = sum(map(lambda x: x >= similarity_score, sco))
    st.markdown("#")
    st.header("Recommendations")
    st.markdown("#")
    if sim_score != 0:
        if sim_score % 3 == 0:
            paletes = int(sim_score/3)
        else:
            paletes = int((sim_score/3)+1)
        i = 0
        for seg in range(paletes):
            if int(sim_score) >= i+3:
                load_img_pattern(unique_list[i:i+3], sco[i:i+3], img_id, uni_id, flipkart_commerce, myntra_commerce)
            else:
                print("hello")
                load_img_pattern(unique_list[i:int(sim_score)], sco[i:int(sim_score)], img_id, uni_id, flipkart_commerce, myntra_commerce)
            i = i+3
    else:
        st.warning("No Matching garment found for given similarity score. Please try altering score or try again with new image")
# Function to add input image
# Run the yolov4 on the cropped section
# Extract the features of the input image using inception resnet
# compare the input features with e-commerce feature vectors
# Rank them based on their similarity score and display the image
def app():
    upper_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_features.npy')
    img_ids_upp = np.load('/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_feature_img_ids.npy')
    lower_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Lower_InceptionResnet_features.npy')
    img_ids_low = np.load('/home/ubuntu/capstone/final_inception_features/Lower_InceptionResnet_feature_img_ids.npy')
    full_ex_fea = np.load('/home/ubuntu/capstone/final_inception_features/Full_InceptionResnet_features.npy')
    img_ids_full = np.load('/home/ubuntu/capstone/final_inception_features/Full_InceptionResnet_feature_img_ids.npy')
    flipkart_commerce = pd.read_csv("/home/ubuntu/capstone/Data/flipkart_ecommerce.csv")
    myntra_commerce = pd.read_csv("/home/ubuntu/capstone/Data/myntra_ecommerce.csv")
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #4682B4;
        color: white   
    }
    .big-font {
        font-size:65px !important;
        color: DarkMagenta;
        text-align: center;
        font-weight: bold;
    }
    .search_font {
    font-size:20px;
    font-style: oblique;
    }
    .image_ref {
    text-align: center;
    }
    
    </style>
    """, unsafe_allow_html=True)

    metrics = ['cosine', "euclidean"]
    st.markdown('<p class="big-font">SnapFash</p>', unsafe_allow_html=True)
    st.markdown('<p class="search_font">Image Search</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    st.sidebar.title("Scoring Filters")
    object_det = st.sidebar.slider("Object Detection Score", 30, 100, 70, 5)
    similarity_score = st.sidebar.slider("Similarity Score", 0, 100, 80, 1)
    metric = st.sidebar.radio("Metrics", metrics, 0)
    if metric == 'cosine':
        met_eval = "cosine"
    elif metric == 'euclidean':
        met_eval = "euclidean"
    if uploaded_file!=None:
        image = Image.open(uploaded_file)
        (w,h) = image.size
        col_up = st.beta_columns(3)
        st.header("Uploaded Image")
        st.write("Make use of the cropper - rectangular box to focus your area of interest")
        cropped_img = st_cropper(image)
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
                    if clas['confidence'] >= (object_det/100):
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
                        print(met_eval)
                        if clas['name'] == 'Upper':
                            distances = scipy.spatial.distance.cdist(upper_ex_fea, query_features, met_eval)
                            results = zip(range(len(distances)), distances)
                            results = sorted(results, key=lambda x: x[1])
                            display_image(results,img_ids_upp,similarity_score,flipkart_commerce,myntra_commerce,met_eval)
                        elif clas['name'] == 'Lower':
                            distances = scipy.spatial.distance.cdist(lower_ex_fea, query_features, met_eval)
                            results = zip(range(len(distances)), distances)
                            results = sorted(results, key=lambda x: x[1])
                            display_image(results, img_ids_low,similarity_score,flipkart_commerce,myntra_commerce,met_eval)
                        elif clas['name'] == 'Full':
                            distances = scipy.spatial.distance.cdist(full_ex_fea, query_features, met_eval)
                            results = zip(range(len(distances)), distances)
                            results = sorted(results, key=lambda x: x[1])
                            display_image(results, img_ids_full,similarity_score,flipkart_commerce,myntra_commerce,met_eval)
                    else:
                        st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")