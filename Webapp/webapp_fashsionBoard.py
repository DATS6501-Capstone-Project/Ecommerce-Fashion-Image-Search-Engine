import streamlit as st
from PIL import Image
import os
import numpy as np
import random



# Function to display the images from the category selected

def garment_cat(random_list, img_ids_cat,cols):
    for i in random_list[:5]:
        cols[0].image(img_ids_cat[i], use_column_width=True)

    for i in random_list[5:10]:
        cols[1].image(img_ids_cat[i], use_column_width=True)

    for i in random_list[10:]:
        cols[2].image(img_ids_cat[i], use_column_width=True)
# Function to select random images from the list of id's to display on the webapp
def app():
    img_ids_upp = np.load('/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_feature_img_ids.npy')
    img_ids_low = np.load('/home/ubuntu/capstone/final_inception_features/Lower_InceptionResnet_feature_img_ids.npy')
    img_ids_full = np.load('/home/ubuntu/capstone/final_inception_features/Full_InceptionResnet_feature_img_ids.npy')
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

    gar_type = st.sidebar.selectbox("Type Selection", ['Upper Garments', 'Lower Garments', 'Full Garments'], 2)

    st.markdown('<p class="big-font">SnapFash</p>', unsafe_allow_html=True)

    cols = st.beta_columns(3)

    if gar_type == 'Upper Garments':
        random_list = random.sample(range(0,img_ids_upp.shape[0]),15)
        garment_cat(random_list,img_ids_upp,cols)

    if gar_type == 'Lower Garments':
        random_list = random.sample(range(0, img_ids_low.shape[0]), 15)
        garment_cat(random_list, img_ids_low,cols)

    if gar_type == 'Full Garments':
        random_list = random.sample(range(0, img_ids_full.shape[0]), 15)
        garment_cat(random_list, img_ids_full,cols)
















