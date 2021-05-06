# Ecommerce-Fashion-Image-Search-Engine

Repository for image-based fashion/garment recommendation engine created for the M.S. capstone course in Data Science at GWU. Includes code for:

### 1. Object Detection Training
Contains two deep learning approaches (Modified VGG and YOLO4.0[darknet]) to object detection and bounding box estimation. Both models trained on the open-source DeepFashion dataset (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). 

### 2. Feature Extraction
Contains garment detection, feature extraction using a variety of pre-trained models (Inception-ResNet, VGG, MobileNet,...), and cosine similarity implementations on open-source e-commerce listings. Garment listings are pooled from Flipkart and Myntra (Indian e-commerce stores), which are used for recommendations. 

### 3. Similarity 
Contains examples of garment recommendations used in manual evaluations to determine the best pre-trained model. For example:
![](evaluations/Inception_res_ex1.jpg?raw=true)
### 4. Webapp
Contains streamlit (https://streamlit.io/) implementation of the fashion recommendation engine. Allows users to upload image and crop the garment of interest. 

## Package installations
1. Tensorflow
2. Keras
3. streamlit
4. streamlit-cropper
5. Darknet Framework - https://github.com/AlexeyAB/darknet
