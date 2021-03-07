import streamlit as st
from PIL import Image
import os
import pandas as pd
from streamlit_cropper import st_cropper
### Excluding Imports ###
st.title("E-Commerce Fashion Visual Search - Category type detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file!=None:
    # img = cv2.imdecode(np.fromstring(uploaded_file, np.uint8), 1)
    # img = cv2.imread(img)
    image = Image.open(uploaded_file)
    (w,h) = image.size # just the way PIL works - cv2 does h,w
    st.header("Uploaded Image.")
    cropped_img = st_cropper(image)
    os.chdir("/home/ubuntu/capstone/check_new/darknet")
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
                st.header("Predicted Image.")
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
                    predict_image = Image.open('predictions.jpg')
                    st.header("Predicted Image.")
                    st.image(predict_image)





