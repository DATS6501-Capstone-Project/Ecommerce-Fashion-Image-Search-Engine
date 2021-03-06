import streamlit as st
from PIL import Image
import os

### Excluding Imports ###
st.title("E-Commerce Fashion Visual Search - Category type detection")

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
    os.chdir("/home/ubuntu/capstone/check_new/darknet")
    with open("./data/test_images.jpg","wb") as f:
        f.write(uploaded_file.getbuffer())
    test_image = 'data/test_images.jpg'
    command_darknet = "./darknet detector test data/obj.data cfg/yolo-obj.cfg yolo-obj_final.weights -dont_show -thresh 0.3 " + test_image
    os.system(command_darknet)
    predict_image = Image.open('predictions.jpg')
    with col2:
        st.header("Predicted Image.")
        st.image(predict_image)
