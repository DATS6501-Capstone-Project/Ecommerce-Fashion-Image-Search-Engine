import streamlit as st
import SessionState
import numpy as np
import os

upper_ids = []
lower_ids = []
full_ids = []

#To get the names of the test images in the folder
for root, dirs, files in os.walk("/home/ubuntu/capstone/darknet/new_test_results/upper_samples", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        upper_ids.append(f)
for root, dirs, files in os.walk("/home/ubuntu/capstone/darknet/new_test_results/lower_samples", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        lower_ids.append(f)
for root, dirs, files in os.walk("/home/ubuntu/capstone/darknet/new_test_results/full_samples", topdown=False):
    for name in files:
        f = os.path.join(root, name)
        full_ids.append(f)

np.save(open('/home/ubuntu/capstone/darknet/new_test_results/eval_upper.npy', 'wb'), np.array(upper_ids))
np.save(open('/home/ubuntu/capstone/darknet/new_test_results/eval_lower.npy', 'wb'), np.array(lower_ids))
np.save(open('/home/ubuntu/capstone/darknet/new_test_results/eval_full.npy', 'wb'), np.array(full_ids))

# Function to fetch the session page number when user click next/previous button
def display_next(img_ids_cat):
    prev, _, next = st.beta_columns([1, 10, 1])
    session_state = SessionState.get(page_number=0)
    last_page = img_ids_cat.shape[0]-1
    if next.button("Next"):

        if session_state.page_number + 1 > last_page:
            session_state.page_number = 0
        else:
            session_state.page_number += 1

    if prev.button("Previous"):

        if session_state.page_number - 1 < 0:
            session_state.page_number = last_page
        else:
            session_state.page_number -= 1

    val = session_state.page_number
    st.image(img_ids_cat[val])

# Function load the results images
def app():
    img_ids_upp = np.load('/home/ubuntu/capstone/darknet/new_test_results/eval_upper.npy')
    img_ids_low = np.load('/home/ubuntu/capstone/darknet/new_test_results/eval_lower.npy')
    img_ids_full = np.load('/home/ubuntu/capstone/darknet/new_test_results/eval_full.npy')
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #4682B4;
          
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
    st.markdown('<p class="big-font">SnapFash</p>', unsafe_allow_html=True)
    st.sidebar.title("Garment Type")
    gar_type = st.sidebar.selectbox("Type Selection", ['Upper Garments', 'Lower Garments', 'Full Garments'], 2)
    if gar_type == 'Upper Garments':
        st.header("Upper Garments Evaluation Samples")
        st.markdown("######")
        display_next(img_ids_upp)

    if gar_type == 'Lower Garments':
        st.header("Lower Garments Evaluation Samples")
        st.markdown("######")
        display_next(img_ids_low)

    if gar_type == 'Full Garments':
        st.header("Full Garments Evaluation Samples")
        st.markdown("######")
        display_next(img_ids_full)





