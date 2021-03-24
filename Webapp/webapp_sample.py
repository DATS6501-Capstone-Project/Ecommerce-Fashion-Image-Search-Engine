import streamlit as st
from PIL import Image
import os
import pandas as pd
from streamlit_cropper import st_cropper
import scipy.spatial
from keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
### Excluding Imports ###
st.title("E-Commerce Fashion Visual Search - Category type detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file!=None:
    # img = cv2.imdecode(np.fromstring(uploaded_file, np.uint8), 1)
    # img = cv2.imread(img)
    image = Image.open(uploaded_file)
    (w,h) = image.size # just the way PIL works - cv2 does h,w
    st.header("Uploaded Image")
    cropped_img = st_cropper(image)
    st.header("Make use of the cropper - rectangular box to focus your area of interest")
    os.chdir("/home/ubuntu/capstone/darknet")
    cropped_img.save("./data/test_images.jpg")
    command_darknet = "./darknet detector test data/obj.data cfg/yolo-obj.cfg yolo-obj_final.weights -dont_show -thresh 0.3 -out result.json < test_sample.txt"
    os.system(command_darknet)
    results = pd.read_json("./result.json")
    for i in results['objects']:
        if len(i) > 1:
            views = 1
            for clas in i:
                if clas['confidence'] <= 0.40:
                    views = 0
            if views != 0:
                predict_image = Image.open('predictions.jpg')
                st.header("Predicted Image")
                st.image(predict_image)
            else:
                st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")
        elif len(i) == 0:
            st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")
        elif len(i) == 1:
            views = 1
            for clas in i:
                if clas['confidence'] <= 0.70:
                    st.warning("Detected Object category exceeds maximum limit or No appropriate product category detected - please try again with new image or modify the crop box to fit exact area of interest")
                else:
                    predict_image = Image.open("/home/ubuntu/capstone/darknet/data/test_images.jpg")
                    center_x = clas['relative_coordinates']['center_x']
                    center_y = clas['relative_coordinates']['center_y']
                    wid = clas['relative_coordinates']['width']
                    heig = clas['relative_coordinates']['height']
                    width = wid/2.0
                    height = heig/2.0
                    img_wid = predict_image.size[0]
                    img_ht = predict_image.size[1]
                    left = (center_x-width)*img_wid
                    right = (center_x+width)*img_wid
                    top = (center_y+height)*img_ht
                    bottom = (center_y-height)*img_ht
                    predict_image = predict_image.crop((left, top, right, bottom))
                    st.header("Predicted Image")
                    st.image(predict_image)

    extracted_features = np.load('/home/ubuntu/capstone/Flipkart_Tops_ResNet_features.npy')
    img_ids = np.load('/home/ubuntu/capstone/Flipkart_Tops_ResNet_feature_img_ids.npy')
    path = "/home/ubuntu/capstone/darknet/data/test_images.jpg"
    model = applications.ResNet50(include_top=False, weights='imagenet')
    img = Image.open(path)
    img = predict_image.resize((224, 224))
    #img = image.load_img(path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    query_features = model.predict(x)
    query_features = np.array(query_features).reshape((1,100352))
    distances = scipy.spatial.distance.cdist(extracted_features, query_features, "cosine")
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    for id,score in results[0:5]:
        print(img_ids[id])
        print(score)
        rec_path = "/home/ubuntu/capstone/Data/Flipkart/Tops/"+img_ids[id]
        recommend_image = Image.open(rec_path)
        st.header("Recommendations")
        st.image(recommend_image)
