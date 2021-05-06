import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np

# Below functions acts as an markdown to add project information on to the webapp
def app():
    # CSS styles
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #4682B4;
        color: white   
    }
    .css-1aumxhk  {
        background-color: Gold;
    }
    .big-font {
        font-size:65px !important;
        color: DarkMagenta;
        text-align: center;
        font-weight: bold;
    }
    .bold_font {
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
    .italic_ref {
    text-align: justify;
    font-style: italic;
    
    }
    .italic_ref1 {
    text-align: justify;
    font-style: italic;
    text-align: center;
    
    }
    ul.a {
  list-style-type: circle;
    }

    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">SnapFash</p>', unsafe_allow_html=True)
    st.markdown('<p class="italic_ref1">An image-based e-commerce fashion recommendation system</p>', unsafe_allow_html=True)
    st.header("Introduction")
    st.markdown("<p>The last decade has brought about a large wave in online retail and shopping. From groceries to electronics, the internet and it’s ever changing e-commerce ecosystem has made everything available at our doorsteps across the world. In the US, e-commerce sales jumped from 5.1% in 2007 to 21% in 2020 of total retail sales. Worldwide, the retail e-commerce market has roughly quadrupled in the last six years. Moreover, with the current pandemic, brick and mortar businesses are becoming less feasible and 2020 saw a huge marketplace shift towards e-commerce platforms.</p>", unsafe_allow_html=True)
    st.markdown("<p>Retrieving the fashion products using image is one of the challenging task in fashion domain. The major Objective of the e-commerce website is to recommend better products by improving the user’s discovery experience. User satisfaction relies on product’s visual appearance. The traditional approach is less effective as they rely on text attributes and description. Also, these approches suffer from the cold start problem when new products are added to the database which carry less information for collborative based recommendation methods. To avoid this, we propose an image-based search can better identify user expectations and recommend products more aligned to those expectations. </p>", unsafe_allow_html=True)
    st.header("Problem Statement")
    st.markdown('<p class="italic_ref">E-commerce fashion recommendations are highly sensitive to user interpretation of the product on the search engine. Text-based input from customers can be vague, and lead to irrelevant suggestions. An image-based recommendation system can better identify customer wants, thereby boosting sales for e-commerce vendors.</p>', unsafe_allow_html=True)
    st.markdown('<p class="bold_font">"A picture is worth a thousand words!"</p>', unsafe_allow_html=True)
    st.header("Problem Elaboration")
    st.markdown(
        "<ul class='a'> <li> With a saturation of e-commerce marketplaces, finding better ways to bridge the gap between customer searches and website recommendations is an increasingly lucrative and challenging task. </li> <li>Shopping for fashion and clothing is a visual experience. Unlike buying a car or a computer, one does not always have exact technical specifications when browsing for clothes. The overall look, pattern and color the user has in mind are often indescribable with words.</li></ul>",
        unsafe_allow_html=True)
    st.header("Project Work Flow")
    st.write("Training Set up")
    path = "/home/ubuntu/capstone/work_flow1.JPG"
    st.write("####")
    st.image(path)
    st.write("SnapFash Real Time Work Flow")
    path = "/home/ubuntu/capstone/work_flow2.JPG"
    st.write("####")
    st.image(path)
    st.header("E-Commerce Data")
    dats = st.selectbox("Database", ['Flipkart', 'Myntra'], 0)
    flipkart_commerce = pd.read_csv("/home/ubuntu/capstone/Data/flipkart_ecommerce.csv")
    myntra_commerce = pd.read_csv("/home/ubuntu/capstone/Data/myntra_ecommerce.csv")
    if dats == "Flipkart":
        st.write(flipkart_commerce[:15])
    else:
        st.write(myntra_commerce[:15])
