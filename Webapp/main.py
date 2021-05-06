# References: https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4

#Load streamlit and all necessary sub pages
import webapp_live
import webapp_fashsionBoard
import webapp_resultBoard
import webapp_introBoard
import webapp_detectionBoard
import streamlit as st
from PIL import Image

title_name = "SnapFash"

# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=title_name,page_icon = "/home/ubuntu/capstone/logo.JPG"
)
PAGES = {
    "Intro Board": webapp_introBoard,
    "Fashion Board": webapp_fashsionBoard,
    "Detection Results Board": webapp_detectionBoard,
    "Results Board": webapp_resultBoard,
    "Play Board": webapp_live
}
st.sidebar.image('/home/ubuntu/capstone/logos.JPG')
st.sidebar.title('Boards')
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app()