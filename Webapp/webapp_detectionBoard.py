import streamlit as st
from PIL import Image
import os
import numpy as np
import random

upper_ids = []
lower_ids = []
full_ids = []

# To get the detected objects lists from the folder
for root, dirs, files in os.walk("/home/ubuntu/capstone/object_detection/upper", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        upper_ids.append(f)
for root, dirs, files in os.walk("/home/ubuntu/capstone/object_detection/lower", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        lower_ids.append(f)
for root, dirs, files in os.walk("/home/ubuntu/capstone/object_detection/full", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        full_ids.append(f)

np.save(open('/home/ubuntu/capstone/object_detection/obj_upp_ids.npy', 'wb'), np.array(upper_ids))
np.save(open('/home/ubuntu/capstone/object_detection/obj_low_ids.npy', 'wb'), np.array(lower_ids))
np.save(open('/home/ubuntu/capstone/object_detection/obj_full_ids.npy', 'wb'), np.array(full_ids))

# Function to display the images on the app
def garment_cat(random_list, img_ids_cat, cols):
    for i in random_list[:5]:
        cols[0].image(img_ids_cat[i], use_column_width=True)

    for i in random_list[5:10]:
        cols[1].image(img_ids_cat[i], use_column_width=True)

    for i in random_list[10:]:
        cols[2].image(img_ids_cat[i], use_column_width=True)

# Load the detected files and randomly select 15 images
def app():
    img_ids_upp = np.load('/home/ubuntu/capstone/object_detection/obj_upp_ids.npy')
    img_ids_low = np.load('/home/ubuntu/capstone/object_detection/obj_low_ids.npy')
    img_ids_full = np.load('/home/ubuntu/capstone/object_detection/obj_full_ids.npy')
    # CSS styles for the page
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

    st.sidebar.title("Garment Type")

    gar_type = st.sidebar.selectbox("Type Selection", ['Upper Garments', 'Lower Garments', 'Full Garments'], 0)

    st.markdown('<p class="big-font">SnapFash</p>', unsafe_allow_html=True)

    cols = st.beta_columns(3)

    if gar_type == 'Upper Garments':
        random_list = random.sample(range(0, img_ids_upp.shape[0]), 15)
        garment_cat(random_list, img_ids_upp, cols)

    if gar_type == 'Lower Garments':
        random_list = random.sample(range(0, img_ids_low.shape[0]), 15)
        garment_cat(random_list, img_ids_low, cols)

    if gar_type == 'Full Garments':
        random_list = random.sample(range(0, img_ids_full.shape[0]), 15)
        garment_cat(random_list, img_ids_full, cols)
















